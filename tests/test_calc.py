import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from multitool import calc
class DummyArgs:
    def __init__(self, a, b):
        self.a = a
        self.b = b

def test_calc(capsys):
    calc.run(DummyArgs(2, 3))
    captured = capsys.readouterr()
    assert '2 + 3 = 5' in captured.out
