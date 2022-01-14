from datetime import datetime
from time import time
from os import mkdir
from os.path import isdir


class Logger:
    """Generates a logger object that allows writing formatted content to a
    logger file.

    Attributes:
    ----------
    ENTRY : str (default '+')
        Character(s) that appear at the beginning of each logged line.
    SEPARATOR : str (default = '->')
        Character(s) that separate the timestamp from the logged content.
    INDENTATION_SPACES : str (default = 4)
        Amount of spaces per indentation level.
    ALIGNMENT_CHARACTER : str (default '-')
        Character used to indent logged contents.
    TIMESTAMP : obj (default = datetime.now().isoformat())
        Timestamp located at the beginning of each logged line.
    DIRECTORY : str (default = 'logs')
        Name of the directory where the log files are stored.
    PREFIX : str (default = 'log_')
        Prefix for each log file's name.
    FORMAT : str (default = 'txt')
        Extension of the log files.
    """

    ENTRY = "+"
    SEPARATOR = "->"
    INDENTATION_SPACES = 4
    ALIGNMENT_CHARACTER = "-"

    DIRECTORY = "logs"
    PREFIX = "log_"
    FORMAT = "txt"

    def __init__(self):
        self.FILE = (
            f"./{self.DIRECTORY}/"
            + f"{self.PREFIX}"
            + f"{''.join(str(time()).split('.'))}."
            + f"{self.FORMAT}"
        )

        if not isdir(self.DIRECTORY):
            mkdir(self.DIRECTORY)

    def log(self, argument, indentation) -> None:
        if not isinstance(argument, (list, tuple, set)):
            argument = [argument]

        # First argument acts as header for the rest (if present).
        argument[0] = (
            f"{self.ENTRY} "
            + f"{datetime.now().isoformat()} "
            + f"{self.ALIGNMENT_CHARACTER * (indentation + 1) * self.INDENTATION_SPACES } "
            + f"{self.SEPARATOR} "
            + f"{argument[0]}\n"
        )

        # The remaining elements are indented and formatted accordingly.
        for index, element in enumerate(argument[1:]):
            argument[index + 1] = (
                f"{self.ENTRY} " + f" {element} ".rjust(
                    len(argument[0]) + self.INDENTATION_SPACES,
                    self.ALIGNMENT_CHARACTER
                ) + "\n"
            )

        argument[-1] += '\n'  # endline extra separation.

        with open(self.FILE, mode='a', encoding="utf-8") as file:
            for element in argument:
                file.write(element)
