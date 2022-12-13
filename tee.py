#!/usr/bin/python
import sys
import traceback


class Tee(object):
    """
    Allow forking of output to stdout and other files
    From: http://stackoverflow.com/questions/11325019/output-on-the-console-and-file-using-python
    @author Thrustmaster <http://stackoverflow.com/users/227884/thrustmaster>
    @author Eric Cousineau <eacousineau@gmail.com>

    Code reference: https://gist.github.com/eacousineau/10427097
    Adapted from this post: https://stackoverflow.com/a/11325249/13976785
    """

    def __init__(self, *files):
        self.files = files

    def open(self):
        """ Redirect stdout """
        if not hasattr(sys, '_stdout'):
            # Only do this once just in case stdout was already initialized
            # @note Will fail if stdout for some reason changes
            sys._stdout = sys.stdout
        sys.stdout = self
        return self

    def close(self):
        """ Restore """
        stdout = sys._stdout
        for f in self.files:
            if f != stdout:
                f.close()
        sys.stdout = stdout

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


# Printi exception with stack trace to file: https://stackoverflow.com/questions/31636884/print-exception-with-stack-trace-to-file/31637063#31637063
# if __name__ == '__main__':
#     print('Start...')
#     try:
#         t = Tee(sys.stdout, open('/tmp/test.txt', 'w')).open()
#         print('Hello world')
#         raise Exception('Test')
#         t.close()
#     except Exception as e:
#         print(e)
#         print(traceback.format_exc())
#         t.close()
#         raise e
#     print('Goodbye')

"""
[ bash ]
$ python tee.py 
Start...
Hello world
Goodbye
$ cat /tmp/test.txt 
Hello world
"""
