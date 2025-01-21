# sample.py


def add_numbers(x, y):
    return x + y  # 格式化不规范，black 应该会自动修正


def subtract_numbers(x, y):
    return x - y  # flake8 应该会检测到这里的注释没有空格
