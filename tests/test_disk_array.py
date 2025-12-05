from qi_align.chunk.disk_array import DiskArray

def test_disk_array_basic():
    arr = DiskArray()
    for i in range(10):
        arr.append(i)

    out = list(arr)
    assert out == list(range(10))
