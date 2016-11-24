#!/usr/bin/env python
# -*- coding: UTF-8 -*-


def getTextToMp3(str, language):
    from gtts import gTTS
    tts = gTTS(text=str, lang=language)
    tts.save("voice.mp3")

def saySomthing(str, language):
    import pygame
    # getTextToMp3("How are you?", 'en')
    # getTextToMp3("你好嗎", 'zh-tw')
    getTextToMp3(str, language)
    pygame.mixer.init()
    pygame.mixer.music.load("./voice.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
    # pygame.mixer.music.stop()
    pygame.mixer.stop()
    pygame.mixer.quit()
    del pygame
    print "End of play"
