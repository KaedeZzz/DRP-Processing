from drp_processing import drp_loader, display_drp, area_mean_drp
from direction import drp_direction_map, get_drp_direction
from utils import ROI

if __name__ == "__main__":
    images, profile = drp_loader()
    # dataset = igrey2drp(images)
    roi = ROI(100, 90, 100, 100)
    mean_drp_mat = area_mean_drp(images, roi, display=True) # display image selected at highest elevation angle
    display_drp(mean_drp_mat)
    print(get_drp_direction(mean_drp_mat))
    drp_direction_map(images, roi)
