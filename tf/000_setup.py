libs = ('cv2', 'tensorflow', 'numpy', 'pandas', 'keras')

for lib in libs:
    exec('import {}; print("{}: ", {}.__version__)'.format(lib, lib, lib))
