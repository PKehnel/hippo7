from glumpy.app import Window


class WindowManager:
    _window_instances = []
    _size = []

    @classmethod
    def get_window(cls, window_num=0) -> Window:
        try:
            return cls._window_instances[window_num]
        except KeyError:
            raise AttributeError(
                "No Window found! You need to set a window with WindowManager.bind_window(window)!"
            )

    @classmethod
    def get_window_size(cls, window_num=0):
        try:
            return cls._size[window_num]
        except KeyError:
            raise AttributeError(
                "No Window found! You need to set a window with WindowManager.bind_window(window)!"
            )

    @classmethod
    def bind_window(cls, window):
        cls._window_instances.append(window)
        cls._size.append((window.width, window.height))

    @classmethod
    def resize(cls, width, height, window_num=0):
        cls._size[window_num] = (width, height)
