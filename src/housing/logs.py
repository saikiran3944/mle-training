import logging


class Logger:
    def __init__(self, filename, message, filemode):
        self.filename = filename
        self.message = message
        self.filemode = filemode

    def logging(self):
        logging.basicConfig(
            filename=self.filename,
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            filemode=self.filemode,
        )
        logger = logging.getLogger()
        logger.debug(self.message)
