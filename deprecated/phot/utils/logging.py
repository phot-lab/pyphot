import logging

logger = logging.getLogger("pyphot")
FORMAT = "%(name)s - [%(pathname)s (line %(lineno)s) - %(funcName)s()] %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)
