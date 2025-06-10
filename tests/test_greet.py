import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from multitool import greet
class DummyArgs:
    def __init__(self, name):
        self.name = name

def test_greet(capsys):
    greet.run(DummyArgs('World'))
    captured = capsys.readouterr()
    assert 'Hello, World!' in captured.out
