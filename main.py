from drp_processing import drp_loader, drp_measure

if __name__ == "__main__":
    images, profile = drp_loader()
    # dataset = igrey2drp(images)
    drp_measure(img_sample=images[17], images=images) # pick image at a highest angle as sample image (to select location from)
