from drp_processing import drp_loader, drp_measure, area_drp
from direction import drp_direction_map, get_drp_direction

if __name__ == "__main__":
    images, profile = drp_loader()
    # dataset = igrey2drp(images)
    roi = [100, 400, 100, 400]
    drp_mat = area_drp(images, roi, display=True) # pick image at a highest angle as sample image (to select location from)
    print(get_drp_direction(drp_mat))
    drp_direction_map(images, roi)
