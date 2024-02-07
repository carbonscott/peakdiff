def split_list_into_chunk(input_list, max_num_chunk = 2):

    chunk_size = len(input_list) // max_num_chunk + 1

    size_list = len(input_list)

    chunked_list = []
    for idx_chunk in range(max_num_chunk):
        idx_b = idx_chunk * chunk_size
        idx_e = idx_b + chunk_size
        if idx_e >= size_list: idx_e = size_list

        seg = input_list[idx_b : idx_e]
        chunked_list.append(seg)

        if idx_e == size_list: break

    return chunked_list
