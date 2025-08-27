from drp_processing import drp_loader, drp_measure, area_DRP

if __name__ == "__main__":
    images, profile = drp_loader()
    # dataset = igrey2drp(images)
    roi = [100, 200, 100, 200]
    area_DRP(images, roi) # pick image at a highest angle as sample image (to select location from)
