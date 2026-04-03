from ao_shaping.utils.file import Recorder

def test_add_record():
    recorder1 = Recorder("mark1", "target1")
    recorder2 = Recorder("mark1", "target1")
    recorder1.append({"_id": 1, "mark1": 0.1, "target1": 0.2})
    recorder2.append({"_id": 2, "mark1": 0.3, "target1": 0.4})
    recorder1 += recorder2
    assert len(recorder1) == 2
    assert recorder1[0]["_id"] == 1
    assert recorder1[1]["_id"] == 2

def test_merge_recorder():
    recorder1 = Recorder("mark1", "target1")
    recorder2 = Recorder("mark1", "target1")
    recorder1.append({"_id": 1, "mark1": 0.1, "target1": 0.2})
    recorder2.append({"_id": 2, "mark1": 0.3, "target1": 0.4})
    recorder1 += recorder2
    assert len(recorder1) == 2
    assert recorder1[0]["_id"] == 1
    assert recorder1[1]["_id"] == 2