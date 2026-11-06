"""Вывод в консоль без падений на Windows.

Когда вывод идёт в трубу (`... | Tee-Object`), Python на Windows пишет в кодировке cp1251,
и символы вроде → ≈ ≠ роняют скрипт (UnicodeEncodeError). Переключаем вывод на UTF-8,
а непредставимые символы заменяем, а не падаем.
Чтобы лог в PowerShell читался, там один раз выполнить:
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
"""
import sys


def utf8_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)   # вывод сразу, построчно
        except (AttributeError, ValueError, OSError):
            pass
