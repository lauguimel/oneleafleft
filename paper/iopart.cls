%% iopart.cls — minimal stub for local compilation (IOP compatibility layer)
%% Full class: https://ctan.org/pkg/iopart
%% Replace this file with the official iopart.cls from CTAN for final submission.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iopart}[2024/01/01 IOP stub for local compilation]

\DeclareOption{10pt}{\PassOptionsToClass{10pt}{article}}
\DeclareOption{11pt}{\PassOptionsToClass{11pt}{article}}
\DeclareOption{12pt}{\PassOptionsToClass{12pt}{article}}
\DeclareOption{letter}{\PassOptionsToClass{letterpaper}{article}}
\DeclareOption{a4paper}{\PassOptionsToClass{a4paper}{article}}
\DeclareOption*{}  % ignore unknown options
\ProcessOptions\relax

\LoadClass[12pt,a4paper]{article}

% Packages
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{geometry}
\geometry{a4paper, margin=2.5cm}
\RequirePackage{amsmath,amssymb}
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\RequirePackage{booktabs}
\RequirePackage{setspace}
\onehalfspacing

% IOP title macros
\newcommand{\ioptwocol}{}
\newcommand{\iopflushbottom}{}
\newenvironment{abstract*}{\begin{abstract}}{\end{abstract}}

% ERL-style running title (stub)
\newcommand{\shorttitle}[1]{}
\newcommand{\shortauthor}[1]{}

% Address / affiliation
\newcommand{\address}[1]{\footnote{#1}}
\newcommand{\ead}[1]{\footnote{E-mail: #1}}

\endinput
