import sys,time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox,
    QTextEdit, QProgressBar, QLabel, QHBoxLayout, QPushButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
from BenchmarkTool import BenchmarkTool,calculateRecall,getLatencyPercentile
from PyQt5.QtWidgets import QFileDialog, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtCore import QThread, pyqtSignal
import PyCANDY as candy
class BenchmarkThread(QThread):
    progress_signal = pyqtSignal(float)  # 用于传递进度
    log_signal = pyqtSignal(str)  # 用于传递日志信息
    status_signal = pyqtSignal(str)  # 用于传递日志信息
    def __init__(self, benchmark_tool, config_map, parent=None,windowObj=None):
        super().__init__(parent)
        self.benchmark_tool = benchmark_tool
        self.config_map = config_map
        self.windowObj = windowObj
        self.resultTensor= None
        self.totalStages = 2
        self.doneProgress = 0
    def setProgressBar(self,value):
        self.progress_signal.emit(value)
    def getQueryResult(self):
        return self.resultTensor
    def run(self):
        # 模拟长时间的基准测试逻辑
        batchSize = int(self.config_map.get("batchSize", 1000))
        eventRate = int(self.config_map.get("eventRate", 1000))
        deleteRows = int(self.config_map.get("deleteRows", -1))
        querySize = int(self.config_map.get("querySize", 100))
        self.querySize = querySize
        self.deleteRows = deleteRows
        annk = int(self.config_map.get("ANNK", 10))
        print(f" Batch size = {batchSize},event rata = {eventRate}, annk = {annk}")
        self.log_signal.emit("Benchmark started...")
        queryTensor, dataTensorInitial, dataTensorStream = self.benchmark_tool.getQueryAndDataTensors()
        self.benchmark_tool.loadInitial()
        self.log_signal.emit("Done loading initial tensor")
        # 删除向量处理
        if deleteRows > 0:
            self.log_signal.emit("Start delete process")
            deleteTensor = dataTensorInitial[:deleteRows]
            self.eventTimestampsDelete = self.benchmark_tool.generateTimestamps(deleteTensor.size(0), eventRate)
            self.processedTimeStampsDelete=self.benchmark_tool.deleteBatchProcess(self.eventTimestampsDelete,deleteTensor,batchSize,self )
            

        # 插入处理
        self.log_signal.emit("Start insert process")
        self.eventTimestampsInsert = self.benchmark_tool.generateTimestamps(dataTensorStream.size(0), eventRate)
        self.processedTimeStampsInsert = self.benchmark_tool.insertBatchProcess(self.eventTimestampsInsert, dataTensorStream, batchSize,self)
        self.log_signal.emit("Run query process")
        start_time = time.time()
        self.resultTensor = self.benchmark_tool.queryProcess(queryTensor,annk)
        self.queryTime = int((time.time() - start_time) * 1e6)  # 转换为微秒
        # 测试完成
        self.log_signal.emit("Benchmark completed.")
    def getLatency(self,percentile):
        latDelete = 0
        if(self.deleteRows>0):
            latDelete = getLatencyPercentile(percentile,self.eventTimestampsDelete,self.processedTimeStampsDelete)
        latInsert = getLatencyPercentile(percentile,self.eventTimestampsInsert,self.processedTimeStampsInsert)
        return latDelete,latInsert
    def genCommonStatistics(self,recall):
        resultDic = {}
        resultDic ['recall'] = float(recall)
        latDelete,latInsert = self.getLatency(0.95)
        resultDic ['95%latency(Insert)'] = float(latInsert)
        resultDic ['95%latency(Del)'] = float(latDelete)
        resultDic['QPS'] = float(self.querySize *1e6) / (self.queryTime)
        return resultDic
class BenchMarkMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up the main window
        self.setWindowTitle("Benchmark Tool")
        self.setGeometry(300, 100, 800, 600)
        self.benchmarkTool = BenchmarkTool() 
        self.refTool = BenchmarkTool() 
        # Initialize the main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Language options
        self.languages = {
            "English": {
                "config": "Select Config",
                "dataset": "Select Dataset",
                "loader": "Select Data Loader",
                "progress": "Progress of Current Step",
                "output": "Intermediate Output",
                "result": "Final Result",
                "start": "Start",
            },
            "中文": {
                "config": "选择配置文件",
                "dataset": "选择数据集",
                "loader": "选择数据加载方式",
                "progress": "当前步骤进度",
                "output": "中间输出",
                "result": "最终结果",
                "start": "开始",
            }
        }
        self.current_language = "English"

        # Language selection
        self.language_selector = QComboBox()
        self.language_selector.addItems(self.languages.keys())
        self.language_selector.currentTextChanged.connect(self.update_language)
        # Progress bar
        self.progress_label = QLabel(self.languages[self.current_language]["progress"])
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Intermediate output area
        self.output_label = QLabel(self.languages[self.current_language]["output"])
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        # Load config file button
        self.load_config_button = QPushButton(self.languages[self.current_language]["config"])
        self.load_config_button.clicked.connect(self.load_config_file)

        # Adding the Load Config button to the main layout
        self.main_layout.addWidget(self.load_config_button)

        # Start button
        self.start_button = QPushButton(self.languages[self.current_language]["start"])
        self.start_button.clicked.connect(self.start_benchmark)

        # Adding the Start button to the main layout
        self.main_layout.addWidget(self.start_button)
        
        

        # Layout management
        self.main_layout.addWidget(self.language_selector)
        
        # Progress layout
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        
        # Output and result layout
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_area)

      

        # Add progress, output, and result to main layout
     # Intermediate output area changed to QTableWidget
        self.output_label = QLabel("Table Output")
        self.output_table = QTableWidget()  # Using QTableWidget for tabular display
        
        # Add to the layout where QTextEdit was used previously
        self.main_layout.addWidget(self.output_label)
        self.main_layout.addWidget(self.output_table)

        self.main_layout.addLayout(progress_layout)
        self.main_layout.addLayout(output_layout)


        # Set the central widget
        self.setCentralWidget(self.main_widget)

    def update_language(self, lang):
        # Update the current language based on selection
        self.current_language = lang
        labels = self.languages[self.current_language]
        
        self.start_button.setText(labels["start"])
        self.progress_label.setText(labels["progress"])
        self.output_label.setText(labels["output"])
        self.load_config_button.setText(labels["config"])
    def on_anns_finished(self):
        """
        处理 BenchmarkThread 完成时的逻辑。
        """
        self.output_area.append("ANNS operation completed. Waiting for thread to finish...")
        self.thread.wait()  # 等待线程完全退出

        self.output_area.append("Start validation phase")
        self.threadVal = BenchmarkThread(self.refTool, self.configMap,windowObj=self)
        self.threadVal.progress_signal.connect(self.setProgressBar)
        self.threadVal.log_signal.connect(self.logOutput)
        self.threadVal.finished.connect(self.on_ref_finished)  # 连接完成信号
        self.threadVal.start()
    def on_ref_finished(self):
        self.output_area.append("Validation completed. Waiting for thread to finish...")
        self.threadVal.wait()  # 等待线程完全退出
        annsResult = self.thread.getQueryResult()
        baselineResult =  self.threadVal.getQueryResult()
        recall = calculateRecall(baselineResult,annsResult)
        self.output_area.append(f"recall={recall}")
        resultDic = self.thread.genCommonStatistics(recall)
        self.displayConfigInTable(resultDic)
    def start_benchmark(self):
        if self.configMap is None:
            self.output_area.append("Please load a config file first.")
            return

        # 创建并启动后台线程
        self.thread = BenchmarkThread(self.benchmarkTool, self.configMap,windowObj=self)
        self.thread.progress_signal.connect(self.setProgressBar)
        self.thread.log_signal.connect(self.logOutput)
        self.thread.start()
        self.thread.finished.connect(self.on_anns_finished)  # 连接完成信号
        

    def setProgressBar(self, value):
        self.progress_bar.setMaximum(10000)
        progress_percentage = int(value * 100)
        self.progress_bar.setValue(progress_percentage)

    def logOutput(self, message):
        self.output_area.append(message)
    def setProgressBar(self,value):
        self.progress_bar.setMaximum(10000)
        progress_percentage = int(value * 100)
        self.progress_bar.setValue(progress_percentage)
    def load_config_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Config File", "", "Config Files (*.csv *.yaml *.xml);;All Files (*)", options=options)
        if file_name:
            self.output_area.append(f"Loaded config file: {file_name}")
            
            self.configMap = self.benchmarkTool.loadConfigFromFile(file_name)
            self.refTool.loadConfigFromFile(file_name,1,'flat')
            self.displayConfigInTable(self.configMap)
           
    def displayConfigInTable(self, config_dict):
        """
        Display the configuration dictionary in QTableWidget format.
        """
        self.output_table.clear()  # Clear existing content if any
        
        # Set table dimensions
        self.output_table.setRowCount(len(config_dict))
        self.output_table.setColumnCount(2)
        self.output_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        
        # Populate table with configuration data
        for row, (key, value) in enumerate(config_dict.items()):
            self.output_table.setItem(row, 0, QTableWidgetItem(key))
            self.output_table.setItem(row, 1, QTableWidgetItem(str(value)))
# Run the application
app = QApplication(sys.argv)
window = BenchMarkMainWindow()
window.show()
sys.exit(app.exec_())
