from drp_processing import drp_loader, igrey2drp, drp_measure

if __name__ == "__main__":
    images, profile = drp_loader()
    # dataset = igrey2drp(images)
    drp_measure(images[0], images=images)
