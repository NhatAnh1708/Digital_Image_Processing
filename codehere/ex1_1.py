from ImageProcessing import ImageProcessing




if __name__ == '__main__':
    ip = ImageProcessing()
    ip.run()
    #ip.showinput()
    #ip.medianfilter()
    #ip.avegaringfilter()
    #ip.morphology_fromclosetoopen()
    ip.morphology_fromopentoclose()

