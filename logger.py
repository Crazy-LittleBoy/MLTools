# -*- coding: utf-8 -*-
import os
import sys
import io
import datetime


def create_log_time():
    now = datetime.datetime.now()
    return str(now).replace(' ', '_').replace(':', '_')


def write_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="default.log", path="./"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(create_log_time() + '.log', path=path)


if __name__ == '__main__':
    write_print_to_file(path="./")
    print('explanation'.center(80, '*'))
    info1 = '从大到小排序'
    info2 = 'sort the form large to small'
    print(info1)
    print(info2)
    print('END: explanation'.center(80, '*'))
