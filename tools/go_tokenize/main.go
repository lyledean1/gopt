package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

var skipDirs = map[string]bool{
	".git":        true,
	"vendor":      true,
	"node_modules": true,
	"third_party": true,
	"dist":        true,
	"build":       true,
}

func main() {
	rootFlag := flag.String("root", "", "root directory containing Go source")
	outFlag := flag.String("out", "data/go/tokens.txt", "output token stream path")
	noFileHeaders := flag.Bool("no-file-headers", false, "disable FILE records")
	preserveValues := flag.Bool("preserve-values", false, "preserve IDENT/literal payloads")
	flag.Parse()

	if *rootFlag == "" {
		fmt.Fprintln(os.Stderr, "--root is required")
		os.Exit(2)
	}

	root, err := filepath.Abs(*rootFlag)
	if err != nil {
		fail(err)
	}

	files, err := collectGoFiles(root)
	if err != nil {
		fail(err)
	}
	if len(files) == 0 {
		fail(fmt.Errorf("no Go files found under %s", root))
	}

	if err := os.MkdirAll(filepath.Dir(*outFlag), 0o755); err != nil {
		fail(err)
	}

	out, err := os.Create(*outFlag)
	if err != nil {
		fail(err)
	}
	defer out.Close()

	writer := bufio.NewWriter(out)
	defer writer.Flush()

	written := 0
	for _, path := range files {
		if err := writeTokenFile(writer, root, path, !*noFileHeaders, *preserveValues); err != nil {
			fail(err)
		}
		written++
	}

	fmt.Printf("tokenized %d Go files to %s\n", written, *outFlag)
}

func collectGoFiles(root string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != root && skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) == ".go" {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	slices.Sort(files)
	return files, nil
}

func writeTokenFile(writer *bufio.Writer, root string, path string, includeFileHeader bool, preserveValues bool) error {
	src, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(src)) == "" {
		return nil
	}

	if includeFileHeader {
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if err := writeRecord(writer, "FILE", filepath.ToSlash(rel)); err != nil {
			return err
		}
	}

	var s scanner.Scanner
	fset := token.NewFileSet()
	file := fset.AddFile(path, fset.Base(), len(src))
	s.Init(file, src, nil, scanner.ScanComments)

	prevLine := 0
	for {
		pos, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}

		if tok == token.COMMENT {
			continue
		}

		line := fset.Position(pos).Line
		if prevLine != 0 && line > prevLine {
			for range line - prevLine {
				if _, err := writer.WriteString("NEWLINE\n"); err != nil {
					return err
				}
			}
		}

		label, value := serializeToken(tok, lit, preserveValues)
		if err := writeRecord(writer, label, value); err != nil {
			return err
		}
		prevLine = line
	}

	_, err = writer.WriteString("NEWLINE\nNEWLINE\n")
	return err
}

func serializeToken(tok token.Token, lit string, preserveValues bool) (string, string) {
	switch tok {
	case token.IDENT:
		if preserveValues {
			return "IDENT", lit
		}
		return "IDENT", ""
	case token.INT:
		if preserveValues {
			return "INT", lit
		}
		return "INT", ""
	case token.FLOAT:
		if preserveValues {
			return "FLOAT", lit
		}
		return "FLOAT", ""
	case token.IMAG:
		if preserveValues {
			return "IMAG", lit
		}
		return "IMAG", ""
	case token.CHAR:
		if preserveValues {
			return "CHAR", lit
		}
		return "CHAR", ""
	case token.STRING:
		if preserveValues {
			return "STRING", lit
		}
		return "STRING", ""
	default:
		return tok.String(), ""
	}
}

func writeRecord(writer *bufio.Writer, label string, value string) error {
	if value != "" {
		quoted, err := json.Marshal(value)
		if err != nil {
			return err
		}
		_, err = writer.WriteString(label + "\t" + string(quoted) + "\n")
		return err
	}
	_, err := writer.WriteString(label + "\n")
	return err
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
