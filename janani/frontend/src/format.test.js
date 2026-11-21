import { describe, expect, it } from 'vitest'
import { emptyStages, extractVideoId, fmtMs, fmtNum, fmtS, IDX_STAGES, QRY_STAGES, splitReasoning } from './format'

describe('extractVideoId', () => {
  const id = 'Ilg3gGewQ5U'
  it.each([
    id,
    `https://www.youtube.com/watch?v=${id}`,
    `https://www.youtube.com/watch?list=PL1&v=${id}&index=3`,
    `https://youtu.be/${id}?si=abc`,
    `https://www.youtube.com/shorts/${id}`,
    `https://www.youtube.com/embed/${id}`,
    `https://m.youtube.com/watch?v=${id}`,
  ])('extracts the id from %s', (url) => expect(extractVideoId(url)).toBe(id))

  it.each(['', 'hello', 'https://vimeo.com/123', 'https://youtube.com/watch?v=short'])('rejects %s', (bad) =>
    expect(extractVideoId(bad)).toBeNull())
})

describe('splitReasoning', () => {
  it('separates finished reasoning', () =>
    expect(splitReasoning('<think>plan</think>Answer')).toEqual({ reasoning: 'plan', answer: 'Answer', thinking: false }))
  it('detects reasoning in progress', () =>
    expect(splitReasoning('<think>still')).toEqual({ reasoning: 'still', answer: '', thinking: true }))
  it('passes plain answers through', () =>
    expect(splitReasoning('Hello')).toEqual({ reasoning: '', answer: 'Hello', thinking: false }))
})

describe('formatters', () => {
  it('formats seconds and milliseconds', () => {
    expect(fmtS(0.25)).toBe('250 ms')
    expect(fmtS(2.5)).toBe('2.50 s')
    expect(fmtS(null)).toBe('—')
    expect(fmtMs(1500)).toBe('1.50 s')
    expect(fmtNum(null)).toBe('—')
  })
  it('builds idle stage maps', () => {
    expect(Object.keys(emptyStages(QRY_STAGES))).toEqual(['qembed', 'mmr', 'context', 'infer', 'answer'])
    expect(emptyStages(IDX_STAGES).fetch).toEqual({ status: 'idle', ms: null, detail: '' })
  })
})

import { fmtClock as _clock, isEmail as _isEmail, maskEmail as _mask, passwordStrength as _strength } from './format'

describe('sign-in helpers', () => {
  it('validates emails', () => {
    expect(_isEmail('me@example.com')).toBe(true)
    expect(_isEmail(' me@example.com ')).toBe(true)
    for (const bad of ['', 'me', 'me@', '@example.com', 'me@example', 'a b@example.com']) expect(_isEmail(bad)).toBe(false)
  })
  it('masks the local part of an email', () => {
    expect(_mask('saketh@gmail.com')).toBe('s•••••@gmail.com')
    expect(_mask('a@x.io')).toBe('a•@x.io')
  })
  it('scores passwords from weak to strong', () => {
    expect(_strength('')).toBe(0)
    expect(_strength('abc')).toBe(1)
    expect(_strength('abcdefgh')).toBeGreaterThanOrEqual(1)
    expect(_strength('Correct-Horse-9-Battery')).toBe(4)
  })
  it('formats the resend / expiry countdown', () => {
    expect(_clock(0)).toBe('0:00')
    expect(_clock(45)).toBe('0:45')
    expect(_clock(600)).toBe('10:00')
  })
})

import { displayName as _name } from './format'

describe('displayName', () => {
  it('derives a friendly first name from an email', () => {
    expect(_name('saketh.pv@example.com')).toBe('Saketh')
    expect(_name('PVSAKETH1@gmail.com')).toBe('Pvsaketh')
    expect(_name('ana_maria@x.io')).toBe('Ana')
  })
  it('falls back gracefully', () => {
    expect(_name('')).toBe('there')
    expect(_name('1234@x.io')).toBe('there')
    expect(_name(undefined)).toBe('there')
  })
})
