#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rodrigo Teles Hermeto

class Cell:

    def __init__(self, timeslot, channel):
        self.timeslot = timeslot
        self.channel = channel
        self.sender = None
        self.receivers = []
        self.used = False

    def is_the_last_cell(self):
        if self.timeslot == 100:
            return True
        return False

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])


class Slotframe:
    def __init__(self):
        self.cells = []
        for i in range(0, 101):
            self.cells.append(Cell(i, 0))  # just shared cell
