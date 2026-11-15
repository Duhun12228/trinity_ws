"""Dashboard-style console output of the master node (plain print, optional ANSI colours)."""
import sys
import time

from master.monitor import LOST, NONE, OK, SLOW

WIDTH = 64

_COLORS = {
    'green': '\033[32m', 'red': '\033[31m', 'yellow': '\033[33m',
    'cyan': '\033[36m', 'bold': '\033[1m', 'dim': '\033[2m',
}
_RESET = '\033[0m'

STATE_COLOR = {OK: 'green', SLOW: 'yellow', LOST: 'red', NONE: 'red'}


class Console:

    def __init__(self, color=True):
        self.color = color

    def c(self, text, *styles):
        if not self.color or not styles:
            return text
        return ''.join(_COLORS[s] for s in styles) + text + _RESET

    def line(self, text=''):
        print(f'{self.c(time.strftime("%H:%M:%S"), "dim")} {text}', flush=True)
        sys.stdout.flush()

    def rule(self, char='=', title=''):
        if title:
            side = max(2, (WIDTH - len(title) - 2) // 2)
            text = f'{char * side} {title} {char * side}'
            text += char * (WIDTH - len(text))
        else:
            text = char * WIDTH
        self.line(self.c(text, 'bold'))

    def state(self, state):
        return self.c(f'{state:<4}', STATE_COLOR[state], 'bold')

    def mode(self, mode):
        return self.c(f'[{mode.value.upper()}]', 'cyan', 'bold')

    def status_row(self, watch, st):
        hz = f'{st.hz:5.1f} Hz' if st.hz > 0 else '   -- Hz'
        self.line(f' {watch.name:<12} {watch.topic:<36} {self.state(st.state)} {hz}')

    # ------------------------------------------------------------ blocks
    def system_check(self, watches, statuses, mode, waited, quiet):
        self.rule('=', 'SYSTEM CHECK')
        for w, st in zip(watches, statuses):
            self.status_row(w, st)
        self.rule('-')
        bad = [(w, st) for w, st in zip(watches, statuses) if st.state != OK]
        if not bad:
            self.line(self.c(f' ALL OK ({waited:.1f} s)', 'green', 'bold'))
        else:
            self.line(self.c(f' {len(bad)} PROBLEM(S) after {waited:.1f} s', 'red', 'bold'))
            hint = pipeline_hint(watches, statuses)
            if hint:
                self.line(self.c(f' hint: {hint}', 'yellow'))
        if quiet:
            self.line(self.c(' quiet: other node logs -> ~/.ros/log/latest/<node>-*.log '
                             '(quiet:=false to show)', 'dim'))
        self.rule('-')
        self.line(f' >>> INITIAL MODE: {self.mode(mode)}')
        self.rule('=')

    def mode_change(self, old, new, reason):
        self.rule('-')
        self.line(f' {self.c("MODE", "yellow", "bold")}  {self.mode(old)} -> {self.mode(new)}'
                  f'   (reason: {reason})')
        self.rule('-')

    def mission_change(self, name, old, new, reason):
        color = {'ARMED': 'yellow', 'REQUEST': 'yellow', 'RUNNING': 'cyan',
                 'DONE': 'green'}.get(new.split('/')[0], 'dim')
        why = f'  ({reason})' if reason else ''
        self.line(f' {self.c("MISSION", "bold")} {name}: {old} -> {self.c(new, color, "bold")}'
                  f'{why}')

    def topic_change(self, watch, old, st):
        hz = f' ({st.hz:.1f} Hz)' if st.hz > 0 else ''
        tag = self.c('RECOVERED', 'green', 'bold') if st.state == OK else \
            self.c('PROBLEM', STATE_COLOR[st.state], 'bold')
        self.line(f' {tag}  {watch.name} {watch.topic}: {old} -> {self.state(st.state)}{hz}')

    def info(self, text):
        self.line(f' {text}')

    def warn(self, text):
        self.line(self.c(f' WARN  {text}', 'yellow'))

    def summary(self, mode, watches, statuses, cmd):
        parts = [self.mode(mode)]
        for w, st in zip(watches, statuses):
            val = f'{st.hz:.0f}Hz' if st.state == OK else st.state
            parts.append(f'{w.name} {self.c(val, STATE_COLOR[st.state])}')
        parts.append(f'v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}')
        self.line(' ' + ' | '.join(parts))


def pipeline_hint(watches, statuses):
    """Watches are in pipeline order: name the first broken stage after an OK one."""
    for i, (w, st) in enumerate(zip(watches, statuses)):
        if st.state == OK:
            continue
        if i == 0:
            return f'{w.name} not running? (first stage of the pipeline)'
        prev = watches[i - 1]
        if statuses[i - 1].state == OK:
            return f'{prev.name} OK but {w.name} {st.state}: check the node publishing {w.topic}'
        return None
    return None
