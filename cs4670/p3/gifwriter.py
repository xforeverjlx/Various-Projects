import subprocess
import os
import platform

from imageio import imwrite


class GifWriter(object):

    def __init__(self, temp_format, dest_gif):
        self.command = ['convert']
        if platform.system() == 'Windows':
            self.command = ['magick', 'convert']

        try:
            subprocess.check_call(self.command + ['--help'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            raise Exception('imagemagick is required for gif support')
        except subprocess.CalledProcessError:
            pass

        self.temp_format = temp_format
        self.dest_gif = dest_gif
        self.temp_filenames = []
        self.closed = False

    def append(self, image):
        if self.closed:
            raise Exception('GifWriter is already closed')
        filename = self.temp_format % len(self.temp_filenames)
        self.temp_filenames.append(filename)
        imwrite(filename, image)

    def close(self):
        subprocess.check_call(self.command + ['-delay', '2', '-loop', '0'] +
                              self.temp_filenames + [self.dest_gif])
        for filename in self.temp_filenames:
            os.unlink(filename)
        self.closed = True
