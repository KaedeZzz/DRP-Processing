from processing import drp_loader

def test_drp_loader():
    images, profile = drp_loader()
    assert profile is not None