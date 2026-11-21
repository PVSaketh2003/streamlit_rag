import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import Mascot from './components/Mascot'

describe('Mascot', () => {
  it.each(['idle', 'listen', 'think', 'happy'])('renders the %s mood', (mood) => {
    const html = renderToString(<Mascot mood={mood} size={90} />)
    expect(html).toContain(`m-${mood}`)
    expect(html).toContain('aria-hidden="true"')
  })
  it('wears headphones only while listening and shows thought dots only while thinking', () => {
    expect(renderToString(<Mascot mood="listen" />)).toContain('rx="9"')
    expect(renderToString(<Mascot mood="idle" />)).not.toContain('rx="9"')
    expect(renderToString(<Mascot mood="think" />)).toContain('m-dots')
    expect(renderToString(<Mascot mood="idle" />)).not.toContain('m-dots')
  })
})
