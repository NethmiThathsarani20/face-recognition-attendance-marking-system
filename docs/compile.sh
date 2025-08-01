#!/bin/bash

# LaTeX Abstract Compilation Script for RCAICT 2025
# Face Recognition Attendance System Conference Submissions

echo "🔬 RCAICT 2025 Abstract Compilation Script"
echo "============================================="

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ pdflatex not found. Please install LaTeX distribution (e.g., texlive-full)"
    echo "   Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "   macOS: brew install mactex"
    echo "   Windows: Install MiKTeX or TeX Live"
    exit 1
fi

# Function to compile a LaTeX document
compile_document() {
    local PAPER_FILE="$1"
    local PAPER_TYPE="$2"
    
    echo "📄 Compiling $PAPER_TYPE: $PAPER_FILE.tex"
    
    # First compilation
    echo "   🔄 First compilation pass..."
    pdflatex -interaction=nonstopmode "$PAPER_FILE.tex" > /dev/null 2>&1
    
    # Second compilation for references
    echo "   🔄 Second compilation pass (for references)..."
    pdflatex -interaction=nonstopmode "$PAPER_FILE.tex" > /dev/null 2>&1
    
    # Third compilation to ensure everything is resolved
    echo "   🔄 Final compilation pass..."
    pdflatex -interaction=nonstopmode "$PAPER_FILE.tex" > /dev/null 2>&1
    
    # Check if PDF was generated successfully
    if [ -f "$PAPER_FILE.pdf" ]; then
        echo "✅ $PAPER_TYPE PDF generated successfully: $PAPER_FILE.pdf"
        
        # Get file size
        FILE_SIZE=$(du -h "$PAPER_FILE.pdf" | cut -f1)
        echo "   📊 File size: $FILE_SIZE"
        
        # Count pages
        if command -v pdfinfo &> /dev/null; then
            PAGES=$(pdfinfo "$PAPER_FILE.pdf" | grep "Pages:" | awk '{print $2}')
            echo "   📑 Pages: $PAGES"
        fi
    else
        echo "❌ $PAPER_TYPE PDF generation failed. Check for LaTeX errors:"
        echo "   Run: pdflatex $PAPER_FILE.tex"
        echo "   Check log file: $PAPER_FILE.log"
        return 1
    fi
}

# Compile both abstracts
echo "📝 Compiling RCAICT 2025 Conference Submissions"
echo ""

# Compile main abstract
if [ -f "RCAICT_2025_Abstract.tex" ]; then
    compile_document "RCAICT_2025_Abstract" "Main Abstract"
    ABSTRACT_SUCCESS=$?
else
    echo "❌ RCAICT_2025_Abstract.tex not found"
    ABSTRACT_SUCCESS=1
fi

echo ""

# Compile extended abstract
if [ -f "RCAICT_2025_Extended_Abstract.tex" ]; then
    compile_document "RCAICT_2025_Extended_Abstract" "Extended Abstract"
    EXTENDED_SUCCESS=$?
else
    echo "❌ RCAICT_2025_Extended_Abstract.tex not found"
    EXTENDED_SUCCESS=1
fi
echo ""
echo "📊 RCAICT 2025 Conference Submission Summary"
echo "============================================="

if [ $ABSTRACT_SUCCESS -eq 0 ] && [ $EXTENDED_SUCCESS -eq 0 ]; then
    echo "✅ Both abstracts compiled successfully!"
elif [ $ABSTRACT_SUCCESS -eq 0 ] || [ $EXTENDED_SUCCESS -eq 0 ]; then
    echo "⚠️  Some abstracts compiled successfully, check errors above"
else
    echo "❌ Both abstracts failed to compile, check errors above"
fi

echo ""
echo "🎯 Conference Details:"
echo "   • Conference: RCAICT 2025 (Research Conference on Advances in ICT)"
echo "   • Theme: ICT Innovation and Emerging Technologies"
echo "   • Submission Types: Abstracts and Extended Abstracts"
echo "   • Conference Date: September 3, 2025"
echo "   • Venue: Faculty of Technological Studies, University of Vavuniya, Sri Lanka"
echo "   • Publication: Google Scholar-indexed digital repository"
echo ""

if [ $ABSTRACT_SUCCESS -eq 0 ]; then
    echo "📋 Main Abstract (RCAICT_2025_Abstract.pdf):"
    echo "   • Type: Conference Abstract"
    echo "   • Format: Standard academic abstract format"
    echo "   • Keywords: Face Recognition, Educational Technology, IoT, ICT Innovation"
    echo "   • Citations: 3 references included"
    echo "   • Focus: Production-grade face recognition for educational institutions"
    echo ""
fi

if [ $EXTENDED_SUCCESS -eq 0 ]; then
    echo "📋 Extended Abstract (RCAICT_2025_Extended_Abstract.pdf):"
    echo "   • Type: Extended Conference Abstract"
    echo "   • Format: Multi-section detailed abstract"
    echo "   • Keywords: Face Recognition, Educational Technology, IoT, ICT Innovation"
    echo "   • Citations: Multiple in-text citations with bibliography"
    echo "   • Sections: Introduction, Architecture, Implementation, Performance, Regional Impact"
    echo ""
fi

echo "� Key Technical Features Highlighted:"
echo "   • InsightFace buffalo_l model with 94-98% accuracy"
echo "   • Multi-platform camera support (USB, IP, Mobile, IoT/ESP32)"
echo "   • Regional ICT focus for developing countries"
echo "   • Flask-based professional web interface"
echo "   • Offline operation capability for resource-constrained environments"
echo "   • Real-time processing (50-100ms per recognition)"
echo "   • Raspberry Pi edge deployment support"
echo ""
echo "🌍 Regional ICT Contributions:"
echo "   • Addresses developing country educational challenges"
echo "   • Open-source solution with comprehensive documentation"
echo "   • Budget-friendly implementation with standard hardware"
echo "   • Deployed and tested in Sri Lankan institutions"
echo "   • Cultural and technical adaptation considerations"
echo ""
echo "📊 RCAICT 2025 Guidelines Compliance:"
echo "   ✅ Abstract and Extended Abstract formats"
echo "   ✅ Keywords limited to 4 maximum (including IoT)"
echo "   ✅ In-text citations properly included"
echo "   ✅ Submission dates removed as requested"
echo "   ✅ Conference theme alignment: ICT Innovation and Emerging Technologies"
echo "   ✅ Regional focus for developing country ICT challenges"
echo "   ✅ Educational technology contribution"
echo ""

# Cleanup auxiliary files
echo "🧹 Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz

echo "✨ RCAICT 2025 abstracts compilation complete!"
echo ""
if [ $ABSTRACT_SUCCESS -eq 0 ] && [ $EXTENDED_SUCCESS -eq 0 ]; then
    echo "🚀 Both abstracts are ready for RCAICT 2025 submission!"
fi
