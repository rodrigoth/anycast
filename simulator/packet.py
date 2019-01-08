#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rodrigo Teles Hermeto

class Packet:

    def __init__(self, node, payload):
        self.node = node
        self.payload = payload
        self.attempts_left = 3
        self.sent_asn = 0
        self.received_asn = 0
        self.is_forwarding = False

    def __str__(self):
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

