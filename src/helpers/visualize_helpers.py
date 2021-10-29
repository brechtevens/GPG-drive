from pyglet import resource


def centered_image(filename):
    """ Returns a centered image from a given image file

    Parameters
    ----------
    filename : str
        the location of the given image
    """
    img = resource.image(filename)
    img.anchor_x = img.width/2.
    img.anchor_y = img.height/2.
    return img