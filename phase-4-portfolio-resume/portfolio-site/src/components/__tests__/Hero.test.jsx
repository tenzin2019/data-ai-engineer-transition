import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Hero from '../Hero'

describe('Hero Component', () => {
  it('renders hero section with main heading', () => {
    render(<Hero />)
    
    // Check if the main heading is present
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument()
  })

  it('renders call-to-action buttons', () => {
    render(<Hero />)
    
    // Check if CTA buttons are present
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeGreaterThan(0)
  })

  it('displays developer information', () => {
    render(<Hero />)
    
    // Check if role/title is displayed
    expect(screen.getByText(/data scientist/i)).toBeInTheDocument()
  })
}) 