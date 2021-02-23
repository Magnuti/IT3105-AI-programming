import yaml


class Arguments:
    def parse_arguments(self):
        raise NotImplementedError()

    def __str__(self):
        x = "Arguments: {\n"
        for key, value in self.__dict__.items():
            x += "\t{}: {}\n".format(key, value)
        x += "}"
        return x


if __name__ == "__main__":
    arguments = Arguments()
    arguments.parse_arguments()
    print(arguments)
