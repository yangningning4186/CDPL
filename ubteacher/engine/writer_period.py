def resolve_writer_period(max_iter, default_period=20):
    max_iter = int(max_iter)
    default_period = int(default_period)
    if max_iter <= 0:
        return 1
    return max(1, min(default_period, max_iter))

