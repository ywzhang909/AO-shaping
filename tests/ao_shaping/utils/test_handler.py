"""Tests for ao_shaping.utils.handler — Register decorator class."""

from ao_shaping.utils.handler import Register


class TestRegister:
    def test_register_and_get(self):
        reg = Register()

        @reg.register("add")
        def add(a, b):
            return a + b

        assert reg["add"] is add
        assert reg["add"](2, 3) == 5

    def test_all_funcs(self):
        reg = Register()

        @reg.register("f1")
        def f1():
            return 1

        @reg.register("f2")
        def f2():
            return 2

        funcs = list(reg.all_funcs)
        assert len(funcs) == 2
        assert f1 in funcs
        assert f2 in funcs

    def test_all_names(self):
        reg = Register()

        @reg.register("alpha")
        def alpha():
            pass

        @reg.register("beta")
        def beta():
            pass

        names = list(reg.all_names)
        assert names == ["alpha", "beta"]

    def test_empty_register(self):
        reg = Register()
        assert list(reg.all_funcs) == []
        assert list(reg.all_names) == []

    def test_overwrite_name(self):
        reg = Register()

        @reg.register("x")
        def x_v1():
            return 1

        @reg.register("x")
        def x_v2():
            return 2

        assert reg["x"]() == 2
