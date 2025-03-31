
with open("/Users/max/Desktop/PhD/PhD/single_author_paper/bibliography.bib", "r") as f:
    with open("/Users/max/Desktop/PhD/PhD/single_author_paper/bibliography_cleaned.bib", "w") as g:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith("url") or line.strip().startswith("urldate"):
                continue
            g.write(line)
