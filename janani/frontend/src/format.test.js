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
