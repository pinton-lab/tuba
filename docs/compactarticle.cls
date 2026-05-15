%
% compactarticle.cls -- compact article style used by the TUBA pipeline
% manuscripts (mouse / human / macaque).  Supports the same author-facing
% commands as the previous iopjournal.cls (\articletype, \title, \author,
% \affil, \email, \keywords, abstract environment, \ack, \funding, \data,
% \roles, \suppdata) so the manuscript bodies do not need to change, but
% drops the IOP Publishing branding, sidebar, and journal header.
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{compactarticle}[2026/05/15 Compact article style]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt,a4paper]{article}

\RequirePackage[margin=22mm,headheight=14pt]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage[colorlinks=true,allcolors=blue]{hyperref}

% Diagonal "DRAFT" watermark on every page.  Comment out to disable.
\RequirePackage{draftwatermark}
\SetWatermarkText{DRAFT}
\SetWatermarkScale{6}
\SetWatermarkColor[gray]{0.92}
\SetWatermarkAngle{55}

\setlength{\parindent}{12pt}
\setlength{\parskip}{0pt plus 1pt}

% Single running header: page number followed by "Pinton Lab".  No footer.
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\thepage\quad Pinton Lab}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\renewcommand{\headrule}{{\color{gray}\hrule height \headrulewidth}\vskip-\headrulewidth}

% ---------------------------------------------------------------------------
% Title block.  Commands are typeset in the order the author calls them in
% the document body (no \maketitle), matching the previous IOP class.
% ---------------------------------------------------------------------------

\newcommand{\articletype}[1]{%
  \noindent{\small\sffamily\bfseries\MakeUppercase{#1}}\par
  \vspace{2pt}{\color{gray}\hrule height 0.4pt}\vspace{8pt}}

\renewcommand{\title}[1]{%
  {\noindent\Large\bfseries\raggedright #1\par}\vspace{6pt}}

\renewcommand{\author}[1]{%
  {\noindent\normalsize\raggedright #1\par}\vspace{2pt}}

\newcommand{\affil}[1]{%
  {\noindent\footnotesize\itshape\raggedright #1\par}\vspace{2pt}}

\newcommand{\email}[1]{%
  {\noindent\footnotesize\textbf{E-mail:}~#1\par}\vspace{2pt}}

\newcommand{\keywords}[1]{%
  {\noindent\footnotesize\textbf{Keywords:}~#1\par}\vspace{8pt}}

\renewenvironment{abstract}{%
  {\color{gray}\hrule height 0.4pt}\vspace{4pt}%
  \noindent{\bfseries Abstract.}\enspace\ignorespaces\small}%
  {\par\vspace{4pt}{\color{gray}\hrule height 0.4pt}\vspace{10pt}\normalsize}

% ---------------------------------------------------------------------------
% Section headings: tight, bold; subsection italic.
% ---------------------------------------------------------------------------

\renewcommand\section{\@startsection{section}{1}{\z@}%
   {-2.0ex \@plus -0.5ex \@minus -0.2ex}%
   {0.6ex \@plus 0.2ex}%
   {\normalfont\normalsize\bfseries\raggedright}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
   {-1.5ex \@plus -0.5ex \@minus -0.2ex}%
   {0.4ex \@plus 0.2ex}%
   {\normalfont\normalsize\itshape\raggedright}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
   {-1.2ex \@plus -0.4ex \@minus -0.2ex}%
   {-0.5em \@plus 0.1em}%
   {\normalfont\normalsize\itshape}}

% ---------------------------------------------------------------------------
% Back-matter commands carried over from the IOP class for source-level
% compatibility.  None of the current manuscripts use them, but the cost of
% keeping them is one line each.
% ---------------------------------------------------------------------------

\newcommand{\ack}[1]{\section*{Acknowledgments}#1}
\newcommand{\funding}[1]{\section*{Funding}#1}
\newcommand{\data}[1]{\section*{Data availability}#1}
\newcommand{\roles}[1]{\section*{Author contributions}#1}
\newcommand{\suppdata}[1]{\section*{Supplementary data}#1}

% Smaller, bold-labelled figure / table captions.
\renewcommand{\@makecaption}[2]{\vskip\abovecaptionskip
  \sbox\@tempboxa{\footnotesize\textbf{#1.}~#2}%
  \ifdim \wd\@tempboxa >\hsize
    \raggedright\footnotesize\textbf{#1.}~#2\par
  \else
    \global\@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

% Footnote-size tables (matches previous style).
\let\oldtabular\tabular
\renewcommand{\tabular}{\footnotesize\oldtabular}

\endinput
