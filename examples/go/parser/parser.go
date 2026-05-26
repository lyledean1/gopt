package parser

type Parser struct {
	input string
	pos   int
}

func (p *Parser) Next() byte {
	if p.pos >= len(p.input) {
		return 0
	}

	ch := p.input[p.pos]
	p.pos++
	return ch
}
