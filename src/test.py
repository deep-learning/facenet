import sys

from facenet import get_learning_rate_from_file

if __name__ == '__main__':
    lr = get_learning_rate_from_file(sys.argv[1], int(sys.argv[2]))
    print(lr)
