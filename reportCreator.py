
import datetime
import json
import os
import re
import shutil
import subprocess

from reportlab.lib import colors, enums
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

#Main function is createPdf, takes a parameter that specifies the folder path of the files

#Necessary constant Rename and Color code dictionary
cellTypeRename = {
  'CE': "Tumor/epithelial cells",
  'TIL': "Tumor infiltrating lymphocytes",
  'CEUK': "Unknown cell type",
  'fib': "Fibroblasts",
  'plasma': "Plasma cells",
  'End': "Endothelial cells",
  'nCE': "Normal cells",
  'Mph': "Macrophages",
  'Neu': "Neutrophils",
  'mimi': "Mitotic mimic",
  'MIT': "Mitotic cells",
}
segmentRename = {'SPA': "Space",
  'CT': "Cellular Tumor",
  'ST': "Stroma",
  'FAT': "Fatty tissue",
  'NE': "Necrosis",
  'Other': "Other",
  'Skin': "Skin",
  "BR-D": "Normal/dysplastic breast",}

segmentColorMap = {
  'SPA': (128, 128, 128),
  'CT': (0, 54, 0),
  'ST': (173, 216, 230),
  'FAT': (212, 235, 157),
  'NE': (255, 255, 0),
  'Other': (208, 235, 241),
  'Skin': (255, 42, 42),
  "BR-D": (68, 78, 172),
}



nucleiColorMap= {
  'CE': (0, 255, 255),
  'TIL': (165, 42, 42),
  'End': (255,165,0),
  'plasma': (0, 54, 178),
  'fib': (255, 255, 255),
  'Neu': (160, 32, 240),
  'Mph': (226, 229, 35),
  'nCE': (0, 250, 146),
  'CEUK': (229, 18, 159),
  'mimi': (0, 0, 0),
  'MIT': (247, 25, 226),
}

stilColorMap = {
  "Stromal TILs": (70,110,108),
  "Peri-tumoral TILs": (233,238,59),
  "Intra-tumoral TILs": (115,233,49),
}


#Paragraph styles for all table cells in table
wordStyleBold = ParagraphStyle(
        name='Normal',
        fontName='Times-Bold',
        leading=6,
        fontSize= 7,
    )
wordStyle = ParagraphStyle(
        name='Normal',
        fontName='Times-Roman',
        leading=6,
        fontSize= 7,
    )

class InteractiveTextField(Flowable):
    def __init__(self, text='',name='name',x=15,y=1):
        Flowable.__init__(self)
        self.name = name
        self.text = text
        self.boxsize = 8
        self.x = x
        self.y = y

    def draw(self):
        self.canv.saveState()
        form = self.canv.acroForm
        form.textfield(name=self.name,
                            tooltip=self.text,
                            value=self.text,
                            width=50,
                            height=10,
                            fontSize=6,
                            x=self.x,
                            y=self.y,
                            textColor=colors.black,
                            fontName='Times-Roman',
                            fillColor=colors.white,
                            relative=True,
                            )
        self.canv.restoreState()
        return

#Creating choicebox in table 1,page 1
class InteractiveChoiceBox(Flowable):
    def __init__(self, text='A Box'):
        Flowable.__init__(self)
        self.text = text
        self.boxsize = 12

    def draw(self):
        self.canv.saveState()
        form = self.canv.acroForm
        options = [('','None'),('1','1'),('2','2'),('3','3')]
        form.choice(name=self.text,
                            tooltip=self.text,
                            value='None',
                            width=50,
                            height=10,
                            x = 5,
                            y=-10,
                            fontSize=7,
                            textColor=colors.black,
                            fontName='Times-Roman',
                            fillColor=colors.white,
                            relative=True,
                            forceBorder=True,
                            options=options)
        self.canv.restoreState()
        return

class SquareFlowable(Flowable):
    def __init__(self, size=5, color=colors.white):
        self.size = size
        self.color = color

    def wrap(self, width, height):
        return self.size, self.size

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.setStrokeColor(colors.black)
        self.canv.setLineWidth(0.5)
        self.canv.rect(0, 0, self.size, self.size, fill=True)

#if image not found
def load_image_or_empty(image_path):
    if os.path.exists(image_path):
        return image_path
    else:
        return ""


#convert the json to table Array
def convertDictToArray(table, header,fontSize):
    wordStyle.whiteSpace = 'nowrap'
    wordStyle.fontSize = fontSize
    wordStyleBold.fontSize = fontSize
    
    header[0] = [Paragraph(i,wordStyleBold) for i in header[0]]
    for key, value in table.items():
        if key in ('0', '1', '2', '3', '4'):
            header.append([Paragraph(str(val), wordStyle) for val in list(value.values())])
        elif not isinstance(table[key], dict):
            header.append([Paragraph(key, wordStyle)] + [Paragraph(str(round(value, 2)), wordStyle)])
        else:
            header.append([Paragraph(key,wordStyleBold)] + [Paragraph(str(val), wordStyle) for val in list(table[key].values())])

    return header

#Extract mitotic score 
def getMitoticScore(text):
    # Define a regular expression pattern to extract numbers
    pattern = r'\b\d+\b'

    # Use re.findall() to find all matching numbers in the text
    matches = re.findall(pattern, text)

    # Extract the first number (in this case, "4")
    if matches:
        number = matches[0]
        return number
    
#Adjusting column and row height for page 1 tables
def getColumnWidthRowHeight(table,width,height):
    num_cols = len(table[0])
    num_rows = len(table) 
    
    colWidths = [width / num_cols] * num_cols
    rowHeights = [height] * num_rows
    
    return [colWidths, rowHeights]


def find_key_by_value(dictionary, targetValue):
    for key, value in dictionary.items():
        if value == targetValue:
            return key
    return None  # Return None if the value is not found


def addSquaresInTable(table,colWidthTableCell,rowHeightTableCell,firstrenameDictionary,firstColorMap,secondRenameDictionary=None,secondColorMap=None,stilColorMap=None):
    for row, values in enumerate(table):
        for column, value in enumerate(values):
            if type(value) is not Table:
                key = find_key_by_value(firstrenameDictionary, value.text)
                if key is not None:
                    if key in firstColorMap:
                        r, g, b = firstColorMap[key]
                        backgroundColor = colors.Color(red=(r / 255), green=(g / 255), blue=(b / 255))
                        
                        data = Table([[SquareFlowable(size=5, color=backgroundColor), value]],colWidths=colWidthTableCell,rowHeights=rowHeightTableCell)
                        
                        cellStyle = TableStyle([
                            ('FONTNAME', (-1, 1), (-1, -1), 'Times-Bold'),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('ALIGN', (1, 1), (1, 1), 'LEFT'),  
                            ("ALIGN",(0,0),(0,0),'CENTER'),
                        ])
                        data.setStyle(cellStyle)
                        table[row][column] = data

                elif key is None and secondRenameDictionary != None:
                    newKey = find_key_by_value(secondRenameDictionary,value.text)
                    if newKey is not None:
                        if newKey in secondColorMap:
                            r, g, b = secondColorMap[newKey]
                            backgroundColor = colors.Color(red=(r / 255), green=(g / 255), blue=(b / 255))
                            data = Table([[SquareFlowable(size=5, color=backgroundColor), value]],colWidths=colWidthTableCell,rowHeights=rowHeightTableCell)
                            cellStyle =TableStyle([
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('ALIGN', (-1, -1), (-1, -1), 'LEFT'),
                            ('FONTNAME', (-1, 1), (-1, -1), 'Times-Bold'),
                            ("ALIGN",(0,0),(0,0),'CENTER'),
                            ])
                            data.setStyle(cellStyle)
                            table[row][column] = data
                elif stilColorMap != None:
                    if stilColorMap.get(value.text + ' TILs') != None:
                        r, g, b = stilColorMap[value.text + ' TILs']
                        backgroundColor = colors.Color(red=(r / 255), green=(g / 255), blue=(b / 255))
                        data = Table([[SquareFlowable(size=5, color=backgroundColor), value]],colWidths=[10,55],rowHeights=[12])
                        
                        cellStyle = TableStyle([
                            ('FONTNAME', (-1, 1), (-1, -1), 'Times-Bold'),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('ALIGN', (-1, -1), (-1, -1), 'LEFT'),
                            ("ALIGN",(0,0),(0,0),'CENTER'),
    
                        ])
                        data.setStyle(cellStyle)
                        table[row][column] = data
    return table


#Container for table and Image in page 1
def page1TableContainer(image,table,height):
    width = 350
    colWidth,rowHeights = getColumnWidthRowHeight(table,width,height)
    colWidth80 = colWidth[0]*0.90
    colWidthTableCell = [colWidth80*0.10, colWidth80*0.95]
    rowHeightTableCell = rowHeights[0]*0.80

    gridStyle =  (TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (-1, 1), (-1, -1), 'Times-Roman'),
    ('WORDWRAP', (0, 0), (-1, -1), True),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
    restructuredTable = addSquaresInTable(table,colWidthTableCell,rowHeightTableCell,segmentRename,segmentColorMap,cellTypeRename,nucleiColorMap)
    newTable = Table(restructuredTable,colWidths=colWidth,rowHeights=rowHeights)
    newTable.setStyle(gridStyle)
    
    tableAndImage = [[Image(image, width=150, height=100,kind='proportional'),newTable]]
    TableContainer = Table(tableAndImage)
    
    TableContainer.setStyle(TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), #yhaper
    ]))
    
    return TableContainer

def createPdf(folder_path):

    # Get the parent directory
    parentDirectory = os.path.dirname(folder_path)
    pdf_file = "report.pdf"
    
    outfilepath = os.path.join( parentDirectory, pdf_file )
    # name of the pdf file
    # Create a PDF document
    doc = BaseDocTemplate(outfilepath, pagesize=A4)
    elements = []

    with open(os.path.join(folder_path, "stils_celltypeTable.json"), 'r') as f:
        stilCellTable = json.load(f)

    with open(os.path.join(folder_path,"stilsTable.json"), 'r') as f:
        stilTable = json.load(f)

    with open(os.path.join(folder_path,"tbInfo.json"), 'r') as f:
        tbInfo = json.load(f)

    with open(os.path.join(folder_path,"sideDensity.json"), 'r') as f:
        sideDensity = json.load(f)

    with open(os.path.join(folder_path,"vesicularityDensity.json"), 'r') as f:
        vescularityDensity = json.load(f)

    with open(os.path.join(folder_path,"nucleoli_data.json"), 'r') as f:
        nucleoliData = json.load(f)

    with open(os.path.join(folder_path,'1085b10-pt4_caseInfo.json'),'r') as caseData:
        caseInfo = json.load(caseData)

    amaranth_logo = 'ama_logo.png'
    qrCode = os.path.join(folder_path,"qrCode.png")

    #Page 3 info
    sideDensityImage = os.path.join(folder_path,'size.png')
    vescularDensityImage = os.path.join(folder_path,'vesicularity.png')
    nucleoliDataImage = os.path.join(folder_path,'nucleoli.png')
    hpfNucleoliImage = load_image_or_empty(os.path.join(folder_path,'hpf_npCE.png'))

    #Page 4 info
    tbImage = load_image_or_empty(os.path.join(folder_path,'tbImage.png'))

    #Page 1 info
    segment = os.path.join(folder_path,'segment.png')
    
    nuclie_heatmap = os.path.join(folder_path,"nuclei_heatmap.png")  
    with open(os.path.join(folder_path,"table1_summary.json"), 'r') as f:
        histologicalScoreData = json.load(f)

    with open(os.path.join(folder_path,"table2_summary.json"), 'r') as f:
        segmentAreaData = json.load(f)

    with open(os.path.join(folder_path,"table3_summary.json"), 'r') as f:
        cellTypeCountData = json.load(f)

    with open(os.path.join(folder_path,"table4_summary.json"), 'r') as f:
        cellTypePercentData = json.load(f)

    #Page 5 info
    segment_overlay_nuclei_heatmap = os.path.join(folder_path,"segment_overlay_nuclei_heatmap.png") 
    thumbnail = os.path.join(folder_path,"thumbnail.png")
    hpfThumbnail = load_image_or_empty(os.path.join(folder_path,'hpfThumbnail.png')) 
    stils = os.path.join(folder_path,"stils.png")
    
    #Remove empty key
    for outer_key, inner_dict in histologicalScoreData.items():
        histologicalScoreData[outer_key] = {key: value for key, value in inner_dict.items() if key != ''}
        
    histologicalScoreTable = convertDictToArray(histologicalScoreData,[['','AI score','AI translated score','Pathologist score']],6)
    segmentAreaTable = []
    header = [Paragraph(i,wordStyleBold) for i in ['Segment','area in mm²','%area']]
    segmentAreaTable.append(header)
    for key,value in segmentAreaData.items():
        segmentKey = ''
        if key in segmentRename:
            segmentKey = Paragraph(segmentRename[key],wordStyleBold)
        area_in_mm2 = Paragraph(str(round(value['area in mm2'],2)),wordStyle)
        percent_area = Paragraph(str(round(value['%area'],2)),wordStyle)
        # Create a list with the extracted values
        extracted_data = [segmentKey, area_in_mm2, percent_area]
        segmentAreaTable.append(extracted_data)
    
    
    cellTypeCountTable = []
    cellTypeHeader = [Paragraph(i,wordStyleBold) for i in ['Cell type', 'Total count', 'per mm²', 'per 1000 tumor cells']]
    cellTypeCountTable.append(cellTypeHeader)
    for key, values in cellTypeCountData.items():
        # Extract the values for 'Segment', 'Total count', 'per mm2', and 'per 1000 epithelial cells'
        cellTypeKey = values['Celltype']
        if cellTypeKey in cellTypeRename:
            cellTypeKey = Paragraph(cellTypeRename[cellTypeKey],wordStyleBold)
        total_count = Paragraph("{:,}".format(values['Total count']),wordStyle)
        per_mm2 = Paragraph(str(round(values['per mm2'],2)),wordStyle)
        per_1000_epithelial_cells = Paragraph(str(values['per 1000 epithelial cells']),wordStyle)
        # Create a list with the extracted values
        extracted_data = [cellTypeKey, total_count, per_mm2, per_1000_epithelial_cells]
        # Append the list to the result
        cellTypeCountTable.append(extracted_data)

    cellTypePercentTable = []
    cellTypePercentHeader = []
    for key, values in cellTypePercentData.items():
        cellTypePercentHeader = list(values.keys())
        cellTypePercentTable.append(['Cell type']+ cellTypePercentHeader)
        break  # Exit the loop after extracting the header once

    for key,values in cellTypePercentData.items():
        extracted_data = []
        if key in cellTypeRename:
            celltype = cellTypeRename[key]
            extracted_data.append(celltype)
        for i in cellTypePercentHeader:
            extracted_data.append(str(values[i])+'%')
        cellTypePercentTable.append(extracted_data)
    
        
    cellTypePercentTable[0] = [segmentRename.get(header, header) for header in cellTypePercentTable[0] if header != ' Cell type']
    for index,i in enumerate(cellTypePercentTable):
        for columnIndex,j in enumerate(i):
            if index == 0:
                cellTypePercentTable[index][columnIndex] = Paragraph(j,wordStyleBold)
            elif columnIndex == 0:
                cellTypePercentTable[index][columnIndex] = Paragraph(j,wordStyleBold)
            else:
                cellTypePercentTable[index][columnIndex] = Paragraph(j,wordStyle)
        
    legendData = []
    for key in cellTypeCountData.keys():
        if cellTypeCountData[key]['Celltype'] != 'MIT' and cellTypeCountData[key]['Celltype'] != 'mimi':
            legendData.append(['',cellTypeRename[cellTypeCountData[key]['Celltype']]])
    
    histologicalScoreTable[1][3] = Table([[InteractiveChoiceBox('choice1')]])
    histologicalScoreTable[2][3] = Table([[InteractiveChoiceBox('choice2')]])
    histologicalScoreTable[3][3] = Table([[InteractiveChoiceBox('choice3')]])
    histologicalScoreTable[4][3] = Table([[InteractiveTextField(name='Overall',x=5,y=-12)]])
    
    histologyTableContainer = page1TableContainer(thumbnail,histologicalScoreTable,20)
    
    segmentAreaTableContainer = page1TableContainer(segment,segmentAreaTable,15)
    cellTypeCountTableContainer = page1TableContainer(nuclie_heatmap,cellTypeCountTable,18)
    cellTypePercentTableContainer = page1TableContainer(segment_overlay_nuclei_heatmap,cellTypePercentTable,25)

    
    majorContainer = Table([[histologyTableContainer],[segmentAreaTableContainer],[cellTypeCountTableContainer],[cellTypePercentTableContainer]])
    elements.append(majorContainer)
    elements.append(PageBreak())
    
    
    mitoticInfoTable = [['','Mitotic cells']]
    for key,value in cellTypeCountData.items():
        for nestedKey,nestedValue in cellTypeCountData[key].items():
            if cellTypeCountData[key][nestedKey] == 'MIT':
                mitoticInfoTable.append(['Total count',str(cellTypeCountData[key]['Total count'])])
                mitoticInfoTable.append(['per mm²',str(round(cellTypeCountData[key]['per mm2'],2))])
                mitoticInfoTable.append(['per 1000 tumor cells',str(cellTypeCountData[key]['per 1000 epithelial cells'])])
    mitoticScore = getMitoticScore(histologicalScoreData['Mitotic score']['AI score'])
    mitoticAIScore = str(histologicalScoreData['Mitotic score']['AI translated score'])
    mitoticInfoTable.append(['Total mitosis in 10 consecutive HPF',mitoticScore])
    mitoticInfoTable.append(['AI derived score',mitoticAIScore])
    
    # mitoticInfoTable = [[Paragraph(j, wordStyle) if i == 0 else Paragraph(j, wordStyle) for i, j in enumerate(row)] for row in mitoticInfoTable]
    for index,i in enumerate(mitoticInfoTable):
        for columnIndex,j in enumerate(i):
            if index == 0:
                mitoticInfoTable[index][columnIndex] = Paragraph(j,wordStyleBold)
            elif index == len(mitoticInfoTable) - 1:
                mitoticInfoTable[index][columnIndex] = Paragraph(j,wordStyleBold)
            else:
                mitoticInfoTable[index][columnIndex] = Paragraph(j,wordStyle)
    
    mitoticTable = Table(mitoticInfoTable,colWidths=[100,50])
    mitoticTable.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (-1, -1), ( -1,-1), 'Times-Bold'),
        ('FONTNAME', (-1, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE',(0,0),(-1,-1),7),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    mitoticTableContainer = Table([[
        Image(hpfThumbnail, width=150, height=100,kind='proportional') if hpfThumbnail != "" else ""
        ,mitoticTable]],colWidths=[400,250])
    mitoticTableContainer.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(mitoticTableContainer)
    elements.append(Spacer(1,10))
    
    # Specify the filename you want to search for
    mainHpf = r'hpf_(\d+)_mitosis\.png'  # Replace with the filename you're looking for

    hpfLocationDictionary = {}
    # Iterate through the files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if re.match(mainHpf,file):
                hpfNumber = re.findall(mainHpf,file)
                hpfLocationDictionary[hpfNumber[0]] = {}
                file_path = os.path.join(root, file)
                hpfLocationDictionary[hpfNumber[0]][file_path] = []
                for root, dirs, allFiles in os.walk(folder_path):
                    for searchFile in allFiles:
                        if f'HPF_{hpfNumber[0]}.png' in searchFile:
                            subHpf_file_path = os.path.join(root, searchFile)
                            if hpfNumber[0] in hpfLocationDictionary:
                                hpfLocationDictionary[hpfNumber[0]][file_path].append(subHpf_file_path)
    sortedHpfKeys = sorted(hpfLocationDictionary.keys(), key=lambda x: int(x))
    sortedHpfDictionary = {}
    for i in sortedHpfKeys:
        sortedHpfDictionary[i] = hpfLocationDictionary[i]
        
    allHpf = []
    # Define the common table style
    commonTableStyle = [
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center-align content
    ]

    for hpfNumber, mainHpfPath in sortedHpfDictionary.items():
        for mainHpf, subHpfList in mainHpfPath.items():
            images = [Image(image_path, width=15, height=15, kind='proportional') for image_path in subHpfList]

            subHpfTable = [[""] * 5 for _ in range(5)]

            # Calculate columnLength once based on the length of subHpfList
            columnLength = min(4, len(images))

            # Populate subHpfTable with images
            for column in range(5):
                for row in range(5):
                    if columnLength > column:
                        index = row + 5 * column
                        if index < len(images):
                            subHpfTable[row][column] = images[index]

            subHpfTableContainer = Table(subHpfTable, colWidths=[20] * 5, rowHeights=[20] * 5)

            # Apply the common table style
            subHpfTableContainer.setStyle(TableStyle(commonTableStyle))

            firstHpf = Table([[Image(mainHpf, width=150, height=100, kind='proportional'), subHpfTableContainer]])
            allHpf.append(firstHpf)
    
    table_data = []
    num_rows = 5
    num_columns = 2

    # Determine the maximum number of cells to fill in the table
    max_cells = min(num_rows * num_columns, len(allHpf))

    # Iterate through rows and columns to populate the table_data
    for row in range(num_rows):
        row_data = []  # Create a row for each iteration
        for column in range(num_columns):
            index = row * num_columns + column  # Calculate the index in the 'allHpf' list
            if index < max_cells:
                # Get the item from 'allHpf' using the calculated index
                item = allHpf[index]
                cell_data = [item]  # Create a cell with the item
            else:
                cell_data = []  # Empty cell if no item is available
            row_data.append(cell_data)

        table_data.append(row_data)  # Append the row to the table data
    hpfTable = Table(table_data)
    elements.append(hpfTable)
    elements.append(PageBreak())
    
    

    cleanedSideDensity = convertDictToArray(sideDensity,[['Statistics','Value']],7)
    cleanedVescularDensity = convertDictToArray(vescularityDensity,[['Statistics','Value']],7)
    cleanednucleoliData = convertDictToArray(nucleoliData,[['Confidence','Proportion of cells with prominent nucleoli']],7)
    
    nuclearTableStyle= TableStyle([
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
    ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),  # First row, excluding the first cell
    ('FONTSIZE',(0,0),(-1,-1),8),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
])

    tableSideDensity = Table(cleanedSideDensity)
    tableSideDensity.setStyle(nuclearTableStyle)

    tableVescularDensity = Table(cleanedVescularDensity)
    tableVescularDensity.setStyle(nuclearTableStyle)

    tableNucleoliData = Table(cleanednucleoliData)
    tableNucleoliData.setStyle(nuclearTableStyle)

    nucleiScoreText = 'AI derived Nuclear pleomorphism score is 2'

    nucleiImageText = '*In the image above, the yellow boxes represent prominent nucleoli'

    nucleiImageTextStyles = ParagraphStyle(
        name='Normal',
        alignment=enums.TA_CENTER,
        fontName = 'Times-Italic',
        fontSize = 8
    )

    nucleiScoreTextStyles = ParagraphStyle(
        name='Normal',
        alignment=enums.TA_LEFT,
        fontName = 'Times-Bold',
        fontSize = 8,
        leftIndent=10,

    )

    NuclearPageTables = [[Image(sideDensityImage, width=180, height=200,kind='proportional'),
                          Image(vescularDensityImage, width=180, height=200,kind='proportional'),
                          Image(nucleoliDataImage, width=180, height=200,kind='proportional')],
                         [tableSideDensity,tableVescularDensity,tableNucleoliData]]

    elements.append(Paragraph(nucleiScoreText,nucleiScoreTextStyles))

    NuclearPageTablesContainer = Table(NuclearPageTables)
    NuclearPageTablesContainer.setStyle(TableStyle([
        ('VALIGN', (0, 1), (-1, 1), 'TOP')]))
    elements.append(NuclearPageTablesContainer)

    elements.append(Spacer(1,20))


    if(hpfNucleoliImage != ""):
        NucleiImageContainer= Table([[Image(hpfNucleoliImage, width=350, height=350,kind='proportional')],[Paragraph(nucleiImageText,nucleiImageTextStyles)]])
        rowh = len(legendData)
        rowHeights = [15]*rowh
        legendTable = Table(legendData,colWidths=[15,100],rowHeights=rowHeights)
        legendStyle = TableStyle([
        #       ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center-align content
            ('FONTNAME', (0, 0), (-1, -1), 'Times-Bold'),  # Set the font
            ('FONTSIZE',(0,0),(-1,-1),6),
            ('GRID', (0, 0), (0, -1), 1, (0, 0, 0)),
        ])


        for row, values in enumerate(legendData):
            for column, value in enumerate(values):
                key = find_key_by_value(cellTypeRename, value)
                if key is not None:
                    if key in nucleiColorMap:
                        r, g, b = nucleiColorMap[key]
                        backgroundColor = colors.Color(red=(r / 255), green=(g / 255), blue=(b / 255))
                        legendStyle.add('BACKGROUND', (column - 1, row), (column - 1, row),backgroundColor )
        legendTable.setStyle(legendStyle)

        NucleiImageLegendContainer = Table([[NucleiImageContainer, legendTable]])
        NucleiImageLegendContainer.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        elements.append(NucleiImageLegendContainer)


    # elements.append(Paragraph(nucleiImageText,nucleiImageTextStyles))

    elements.append(PageBreak())
    
    data = [
        ['',Paragraph('Tubules', wordStyleBold)],
        [Paragraph('Total no. in CT',wordStyle), Paragraph(str(tbInfo['total_number_tubules_in_ct_10']),wordStyle)],
        [Paragraph('Total percentage in CT',wordStyle), Paragraph(str(round(tbInfo['total_percentage_tubule_area_in_ct'], 2)),wordStyle)],
        [Paragraph('per mm²',wordStyle), Paragraph(str(round(tbInfo['Tubule/mm2'], 2)),wordStyle)],
        [Paragraph('AI derived Tubule/Acinar formation score', wordStyleBold), Paragraph(str(tbInfo['Tubule score']), wordStyleBold)]
    ]

    tubuleTable = Table(data,colWidths=[100,50])
    tubuleTable.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),  # First row, excluding the first cell
        ('FONTSIZE',(0,0),(-1,-1),8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))


    tbImageText = '*In the above image, the red contours represent tubular/acinar structures in CT region and the blue contours represent tubular/acinar structures in non-CT region'
    tbImageTextStyles = ParagraphStyle(
        name='Normal',
        alignment=enums.TA_CENTER,
        fontName = 'Times-Italic',
        fontSize = 8

    )

    tubuleTableContainer = Table([['',tubuleTable]],colWidths=[400,250])

    # elements.append(Paragraph(text,styles))
    elements.append(tubuleTableContainer)
    elements.append(Spacer(1,20))
    if(tbImage != ""):
        tbImageContainer = Image(tbImage, width=500, height=500,kind='proportional')
        elements.append(tbImageContainer)
        elements.append(Paragraph(tbImageText,tbImageTextStyles))

    elements.append(PageBreak())    

    stilTableHeader = [''] + list(stilTable[next(iter(stilTable))].keys())
    stilCellTableHeader = ['no. of cells per mm²']+list(stilCellTable[next(iter(stilCellTable))].keys())
    cleanedStilTable = convertDictToArray(stilTable,[stilTableHeader],7)
    for row,values in enumerate(cleanedStilTable):
        for column,value in enumerate(values):
            if stilColorMap.get(value.text) != None:
                r, g, b = stilColorMap[value.text]
                backgroundColor = colors.Color(red=(r / 255), green=(g / 255), blue=(b / 255))
                data = Table([[SquareFlowable(size=5, color=backgroundColor), value]],colWidths=[15,50],rowHeights=[10])
                cellStyle = TableStyle([
                    ('FONTNAME', (-1, 1), (-1, -1), 'Times-Bold'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ALIGN', (1, 1), (1, 1), 'LEFT'),
                    ("ALIGN",(0,0),(0,0),'CENTER')
                ])
                data.setStyle(cellStyle)
                cleanedStilTable[row][column] = data

    cleanedStilCellTable = convertDictToArray(stilCellTable,[stilCellTableHeader],7)
    table_width = 570 # Adjust as needed to fit within the page
    num_cols = len(cleanedStilCellTable[0])
    colWidth = [table_width / num_cols] * num_cols
    
    # Create the table and set column widths

    # Add style to the table
    style = TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Times-Bold'),  # Make the first column bold
            ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),  # First row, excluding the first cell
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),

    ])
    restructuredStilCellTypeTable = addSquaresInTable(cleanedStilCellTable,[10,52],[15],cellTypeRename,nucleiColorMap,stilColorMap=stilColorMap,secondRenameDictionary=None,secondColorMap=None)
    stilCellTypeArrayTable = Table(restructuredStilCellTypeTable, colWidths=colWidth,rowHeights=[25,25,25,25])
    stilCellTypeArrayTable.setStyle(style)

    
    stilArrayTable = Table(cleanedStilTable)
    stilArrayTableStyle = TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (0, -1), 'Times-Bold'),  # Make the first column bold
        ('FONTSIZE',(0,0),(-1,-1),8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),

    ])
    stilArrayTable.setStyle(stilArrayTableStyle)


    stilTableContainer = [[
            Image(segment_overlay_nuclei_heatmap, width=200, height=100,kind='proportional')],[stilArrayTable]]
    stilTableContainerStyle = TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])


    stilTableContainerTable = Table(stilTableContainer,rowHeights=[0.6*200,0.4*200])
    stilTableContainerTable.setStyle(stilTableContainerStyle)

    wholeBigContainer = [[stilTableContainerTable,Image(thumbnail, width=200, height=200,kind='proportional'),],]

    containerTable = Table(wholeBigContainer,colWidths=[570 / 2, 570 / 2],rowHeights=200)

    containerTableStyle = TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    containerTable.setStyle(containerTableStyle)
    
    elements.append(containerTable)
    elements.append(Spacer(1, 20))  # 20 points of space
    elements.append(stilCellTypeArrayTable)
    elements.append(Spacer(1,20)) 
    stilImage = Image(stils, width=300, height=300,kind='proportional')
    elements.append(stilImage)

    #Header and Footer
    def add_header(canvas, doc):
        # Create a table for patientDetails and set its style
        styles = getSampleStyleSheet()
        header1_style = ParagraphStyle(
        name='Header1',
        parent=styles['Normal'],
        fontSize=16,             
        leading=20,              
        textColor='#9f2b68',     
    #     spaceAfter=12,           
        fontName='Times-Bold' 
    ) 
        
        ID=caseInfo['ID']
        Age=caseInfo['Age']
        Gender=caseInfo['Gender']
        ER=caseInfo['ER']
        PR=caseInfo['PR']
        Her2=caseInfo['Her2']
        Stage=caseInfo['Stage']
        PAM50=caseInfo['PAM50']

        label_style = ParagraphStyle(
        name='LabelStyle',
        fontSize=7,
        fontName='Times-Roman',
        
        )

        patientDetails = [
            [
                
                [Paragraph('<b>ID:</b>', label_style), InteractiveTextField(ID,"ID",10)],
                [Paragraph('<b>Age:</b>', label_style), InteractiveTextField(Age,"Age",14)],
                [Paragraph('<b>Gender:</b>', label_style), InteractiveTextField(Gender,"Gender",25)],
            ],  
            [
                [Paragraph('<b>ER:</b>', label_style), InteractiveTextField(ER,"ER",12)],
                [Paragraph('<b>PR:</b>', label_style), InteractiveTextField(PR,"PR",12)],
                [Paragraph('<b>Her2:</b>', label_style), InteractiveTextField(Her2,"HER2",17)],
                [Paragraph('<b>Stage:</b>', label_style), InteractiveTextField(Stage,"Stage",19)],
                [Paragraph('<b>PAM50:</b>', label_style), InteractiveTextField(PAM50,"PAM50",25)],
            ],
        ]
        pageTitles = ['Summary','Mitosis','Nuclear pleomorphism','Tubular/Acinar formation','Tumor infiltrating lymphocytes']
        pageTitle = pageTitles[canvas.getPageNumber() - 1]
        col_widths = 400 / 5
        table = Table(patientDetails,colWidths=col_widths)
        table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'LEFT'),('FONTSIZE',(0,0),(-1,-1),8)])
                      )
        # Get the width and height of the table
        width, height = table.wrap(doc.width, doc.topMargin)
        # Draw the table in the header area
        table.drawOn(canvas, 20, 800)

        canvas.drawImage(amaranth_logo, 490, 800, width=100, height=40,preserveAspectRatio=True,mask='auto')  # Adjust width and height as needed

        pageTitleText = Paragraph(pageTitle, header1_style)
        width, height = pageTitleText.wrap(doc.width, doc.bottomMargin)
        pageTitleText.drawOn(canvas, 20, 770)

        color = colors.HexColor('#9f2b68')
        canvas.setStrokeColor(color)
        canvas.rect(0,0,doc.width+doc.leftMargin+doc.rightMargin,doc.height+doc.topMargin+doc.bottomMargin, fill=False, stroke=True)


    # Define a function to add the footer
    def add_footer(canvas, doc):
        today_date = datetime.date.today()
        formatted_date = today_date.strftime("%d/%m/%Y")  # Format as YYYY-MM-DD

        styles = ParagraphStyle(
        name='Normal',
        fontName = 'Times-Roman',
        fontSize = 8
        )
        footer_text = f'Date : {formatted_date}'
        footer = Paragraph(footer_text, styles)
        width, height = footer.wrap(doc.width, doc.bottomMargin)
        footer.drawOn(canvas, 20, 10)
        pageNumber = str(canvas.getPageNumber())
        
        link_text = "Go to image viewer"
        link = caseInfo['link']
        
        text = f'<link href="{link}"><u><font color="blue">{link_text}</font></u></link>'

        paragraph = Paragraph(text, styles)
        paragraph.wrap(doc.width, doc.bottomMargin)
        canvas.setFont('Times-Roman',7)
        canvas.drawString(280,10,pageNumber)
        canvas.setFont('Times-Roman',8)
        if canvas.getPageNumber() == 1:
            canvas.drawImage(qrCode, 530, 10, width=50, height=50,preserveAspectRatio=True,mask='auto')
            paragraph.drawOn(canvas,450,10)
        if canvas.getPageNumber() == 2:
            hpfText = '''*HPF area equivalent to diameter of 0.51mm'''
            hpfText1 = 'Score 1: up to 7' 
            hpfText2 = 'Score 2: 8 - 14'  
            hpfText3 ='Score 3: 15 or more'
            canvas.setFont('Times-Italic', 8)
            canvas.drawString(430,40,hpfText)
            canvas.drawString(430,30,hpfText1)
            canvas.drawString(430,20,hpfText2)
            canvas.drawString(430,10,hpfText3)

    page_width, page_height = A4
    left_margin = right_margin = top_margin = bottom_margin = 2  # Default margins in points
    frame_width = page_width - (left_margin + right_margin)
    frame_height = page_height - (top_margin + bottom_margin)

    # Create the frame
    frame = Frame(left_margin, bottom_margin, frame_width, frame_height,topPadding=70,showBoundary=1)

    # Add the PageTemplate to the document
    template = PageTemplate(id='my_template', frames=[frame], onPage=add_header, onPageEnd=add_footer)
    
    doc.addPageTemplates([template])
    doc.build(elements)



def get_ghostscript_path():
    gs_names = ["gs", "gswin32", "gswin64c"]
    for name in gs_names:
        if shutil.which(name):
            return shutil.which(name)
    raise FileNotFoundError(
        f"No GhostScript executable was found on path ({'/'.join(gs_names)})"
    )


def compress_pdf(path):
    createPdf(path)
    gs = get_ghostscript_path()
    parentDirectory = os.path.dirname(path)
    input_file_path = os.path.join(parentDirectory,"report.pdf")
    output_file_path = os.path.join(parentDirectory,"report_compressed.pdf")
    subprocess.call(
        [
            gs,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/prepress",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            "-sOutputFile={}".format(output_file_path),
            input_file_path,
        ]
    )

compress_pdf(r"C:\Users\joash\OneDrive\Documents\ReportImage\report")
