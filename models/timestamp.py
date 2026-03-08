def timestamp_transform(x):
    """
    Input:  B x T x C x H x W
    Output: B x C x T x H x W
    """

    return x.permute(0, 2, 1, 3, 4).contiguous()
