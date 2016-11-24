#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from gtts import gTTS
tts = gTTS(text='Hello', lang='en')
tts.save("hello.mp3")