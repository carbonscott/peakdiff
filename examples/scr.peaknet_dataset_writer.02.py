import os
import yaml
import csv
import json

import ray

import numpy as np
import h5py

from peakdiff.crystfel_stream.data import StreamManager, StreamConfig, StreamPeakDiff, StreamPeakDiffConfig

from bragg_peak_fitter.utils                   import get_patch_list
from bragg_peak_fitter.engine                  import PeakFitter
from bragg_peak_fitter.modeling.pseudo_voigt2d import PseudoVoigt2D

@ray.remote
def process_sub_event(patch, max_nfev = 2000):
    fitting_result = PeakFitter.fit(patch, max_nfev = max_nfev)    # Return a Ray ObjectRef to the future result

    # [[[ Produce a label ]]]
    # Obtain the model profile...
    model_params  = fitting_result.params
    model_profile = PseudoVoigt2D(model_params)

    # Only focus on the pseudo voigt2d profile without the background correction...
    model_profile.includes_bg = False
    H, W = patch.shape[-2:]
    y = np.arange(0, H)
    x = np.arange(0, W)
    Y, X = np.meshgrid(y, x, indexing = 'ij')
    model_no_bg_val = model_profile(Y, X)

    # Re-align the model so that all values are positive...
    model_no_bg_val -= model_no_bg_val.min()

    # Generate a label by selecting pixels that are larger than the half max of the profile...
    threshold_level = 0.5
    model_no_bg_val_half_max = model_no_bg_val >= (model_no_bg_val.max() * threshold_level)

    return model_no_bg_val_half_max, fitting_result


@ray.remote
def process_event_batch(batch_idx, event_batch, shared_data_ref):
    '''
    Each event corresponds to a frame idx.
    '''
    # Unpack the shared data...
    ## shared_data     = ray.get(shared_data_ref)
    basename_h5     = shared_data_ref['basename_h5'    ]
    dir_h5          = shared_data_ref['dir_h5'         ]
    max_nfev        = shared_data_ref['max_nfev'       ]
    sigma_cut       = shared_data_ref['sigma_cut'      ]
    win_size        = shared_data_ref['win_size'       ]
    stream_peakdiff = shared_data_ref['stream_peakdiff']

    results = []
    for event in event_batch:
        # Obtain img and peaks...
        frame_idx = event
        img       = get_img(frame_idx, stream_peakdiff)
        peaks     = get_good_predicted_peaks(frame_idx, stream_peakdiff, sigma_cut = sigma_cut)

        # Get a patch list for this event...
        peaks      = np.array(peaks)
        peaks_y    = peaks[:, 0]
        peaks_x    = peaks[:, 1]
        patch_list = get_patch_list(peaks_y, peaks_x, img, win_size)

        # Submit sub-events (peak fitting)...
        sub_event_futures = [ process_sub_event.remote(patch, max_nfev) for patch in patch_list ]
        sub_event_results = ray.get(sub_event_futures)

        # Unpack the results of all sub events...
        mask_label_sub_events, fitting_result_sub_events = [ list(row) for row in zip(*sub_event_results) ]

        # Prepare the mask...
        mask = np.zeros_like(img, dtype = bool)
        mask_patch_list = get_patch_list(peaks_y, peaks_x, mask, win_size)

        # Assign labeled patches to mask patches...
        for mask_label, mask_patch in zip(mask_label_sub_events, mask_patch_list):
            mask_patch[:] = mask_label[:]

        # Aggregate results for this event
        event_result = {
            'image' : img,
            'mask'  : mask,
        }
        results.append(event_result)

    # Write results to HDF5...
    file_h5 = f"{basename_h5}.{batch_idx:04d}.hdf5"
    path_h5 = os.path.join(dir_h5, file_h5)
    write_results_to_h5(path_h5, results)

    return batch_idx


def write_results_to_h5(path_h5, results):
    '''
    results:
        [(img, mask, bbox, metadata),
         (img, mask, bbox, metadata),
         ...
        ]
    '''
    with h5py.File(path_h5, 'w') as f:
        for enum_idx, result in enumerate(results):
            image = result['image']
            mask  = result['mask' ]

            # Create a subgroup for each image to store its data and metadata
            img_subgroup = f.create_group(f"Image_{enum_idx}")

            # Store the image and mask with chunking and compression
            img_subgroup.create_dataset("image", data=image,
                                                 dtype = 'float32',
                                                 chunks=(480, 480),)
            img_subgroup.create_dataset("mask" , data=mask,
                                                 dtype = 'int',
                                                 chunks=(480, 480),
                                                 compression="gzip",)

            ## # Store bounding boxes as a variable-length dataset
            ## bbox_dset = img_subgroup.create_dataset("bbox", data=bbox
            ##                                                 dtype = 'float32',)

            ## # Store metadata
            ## serialized_meta = json.dumps(meta)
            ## img_subgroup.attrs['metadata'] = serialized_meta
        print(f"Exporting {path_h5} done.")



def get_img(frame_idx, stream_peakdiff):
    return stream_peakdiff.stream_manager.get_img(frame_idx)


def get_good_predicted_peaks(frame_idx, stream_peakdiff, sigma_cut = 1):
    '''
    [(y, x), (y, x), ...]

    These peaks are worth of labeling.
    '''
    return stream_peakdiff.stream_manager.get_predicted_peaks(frame_idx, sigma_cut)


def main():
    ray.init(address='auto')

    # ___/ CONFIG \___
    # Events to process...
    path_csv       = 'peaknet10k.frame_idx.csv'
    frame_idx_list = []
    with open(path_csv, 'r') as fh:
        lines = csv.reader(fh)
        for line in lines:
            if line:
                frame_idx_list.append(int(line[0].strip()))

    batch_size     = 10
    init_num_tasks = 4
    max_nfev       = 2000
    sigma_cut      = 0.2
    win_size       = 4
    basename_h5    = 'test'
    dir_h5         = 'peaknet-10k'

    # Retrieve data source from the peakdiff...
    path_peakdiff_config = 'peakdiff_config.stream.yaml'
    with open(path_peakdiff_config, 'r') as file:
        peakdiff_config = yaml.safe_load(file)
    stream_config          = StreamConfig(**peakdiff_config['stream_config'])
    dir_output             = peakdiff_config['dir_output']
    stream_peakdiff_config = StreamPeakDiffConfig(stream_config = stream_config, dir_output = dir_output)
    stream_peakdiff        = StreamPeakDiff(stream_peakdiff_config)

    # ___/ RAY TASKS \___
    # Save stream peakdiff data into a ray object store...
    shared_data = {
        'basename_h5'     : basename_h5,
        'dir_h5'          : dir_h5,
        'max_nfev'        : max_nfev,
        'sigma_cut'       : sigma_cut,
        'win_size'        : win_size,
        'stream_peakdiff' : stream_peakdiff,
    }
    shared_data_ref = ray.put(shared_data)

    # Create batches of events (one event corresponds to one frame)...
    num_events    = len(frame_idx_list)
    event_batches = [frame_idx_list[i:min(i + batch_size, num_events)] for i in range(0, num_events, batch_size)]

    # [[[ Elastic task scheduling ]]]
    # Submit the init few batch processing tasks...
    process_batch_idx_list = []   # Python list as a queue (fetch data with .pop)
    batch_idx = 0
    for _ in range(min(init_num_tasks, len(event_batches))):
        # Get a batch from the batch queue...
        batch = event_batches.pop(0)

        # Submit this batch...
        process_batch_idx_list.append(process_event_batch.remote(batch_idx, batch, shared_data_ref))

        # Update batch index...
        batch_idx += 1

    # Continuously submit new tasks once an old one is complete (elastic submission)...
    while event_batches or process_batch_idx_list:
        # Wait for one batch to complete...
        done_batch_idx_list, process_batch_idx_list = ray.wait(process_batch_idx_list, num_returns = 1)
        completed_batch_idx = ray.get(done_batch_idx_list[0])

        # If there are more batches??? 
        if event_batches:
            # Submit the batch now...
            batch = event_batches.pop(0)
            process_batch_idx_list.append(process_event_batch.remote(batch_idx, batch, shared_data_ref))
            batch_idx += 1

    ray.shutdown()


if __name__ == "__main__":
    main()
