
class ImageParam(object):
    def __init__(self, th_min: int, th_max: int, th_num: int,
                 ph_min: int, ph_max: int, ph_num: int):
        self.ph_min = ph_min
        self.ph_max = ph_max
        self.ph_num = ph_num
        self.ph_step = (ph_max - ph_min) / (ph_num - 1)
        self.th_min = th_min
        self.th_max = th_max
        self.th_num = th_num
        self.th_step = (th_max - th_min) / (th_num - 1)

    def __str__(self):
        return "Current image set DRP parameters:\n"\
                + "phi_min: " + str(self.ph_min) + "\n"\
                + "phi_max: " + str(self.ph_max) + "\n"\
                + "phi_num: " + str(self.ph_num) + "\n"\
                + "phi_step: " + str(self.ph_step) + "\n"\
                + "th_min: " + str(self.th_min) + "\n"\
                + "th_max: " + str(self.th_max) + "\n"\
                + "th_num: " + str(self.th_num) + "\n"\
                + "th_step: " + str(self.th_step)