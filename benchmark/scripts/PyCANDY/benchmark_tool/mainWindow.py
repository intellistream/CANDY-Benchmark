import sys,time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, 
    QTextEdit, QProgressBar, QLabel, QHBoxLayout, QPushButton, 
    QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, QFormLayout, QFrame,QCheckBox
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
        self.totalStages = 1
        self.doneProgress = 0
    def setProgressBar(self,value):
        if(self.totalStages==2):
            if(self.doneProgress==0):
                displayVal = value/4
            if(self.doneProgress==1):
                displayVal = value/4+25
        else:
            displayVal = value/2
        if(self.benchmark_tool.refFlag()):
            displayVal = displayVal+50
        self.progress_signal.emit(displayVal)
    def getQueryResult(self):
        return self.resultTensor
    def run(self):
        # 模拟长时间的基准测试逻辑
        batchSize = int(self.config_map.get("batchSize", 1000))
        eventRate = int(self.config_map.get("eventRateTps", 1000))
        deleteRows = int(self.config_map.get("deleteRows", -1))
        querySize = int(self.config_map.get("querySize", 100))
        vecVolume = int(self.config_map.get("memBufferSize",10000 ))
        memBufferSize = int(self.config_map.get("memBufferSize",vecVolume ))
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
            self.totalStages = 2 
            deleteTensor = dataTensorInitial[:deleteRows]
            self.eventTimestampsDelete = self.benchmark_tool.generateTimestamps(deleteTensor.size(0), eventRate)
            self.processedTimeStampsDelete=self.benchmark_tool.deleteBatchProcess(self.eventTimestampsDelete,deleteTensor,batchSize,self )
            self.doneProgress = 1
            

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
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, 
    QTextEdit, QProgressBar, QLabel, QHBoxLayout, QPushButton, 
    QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, QFormLayout, QFrame
)
from PyQt5.QtCore import Qt

class BenchMarkMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Benchmark Tool")
        self.setGeometry(300, 100, 800, 600)
        self.benchmarkTool = BenchmarkTool()
        self.refTool = BenchmarkTool()

        # Multi-language options
        self.languages = {
            "English": {
                "config": "Select Config",
                "dataset_file": "Select Data Vector File",
                "query_file": "Select Query Vector File",
                "enable_random": "Enable Built-in Random Dataset",
                "vec_volume": "Total vector volume",
                "vec_dim": "Vector Dimension",
                "initial_size": "Initial Data Vector Size",
                "delete_size": "Delete Vector Size",
                "insert_size": "Insert Vector Size",
                "query_size": "Query Vector Size",
                "annk": "K Value for Query",
                "batch_size": "Batch Size",
                "start": "Start",
                "progress": "Progress of Current Step",
                "output": "Intermediate Output",
                "result": "Final Result",
                "toggle_output": "Show/Hide Intermediate Output",
                "index_tag":"Name of Algo",
                "cuda_idx":"Cuda device id",
                "event_rate":"Event Rate"
            },
            "中文": {
                "config": "选择配置文件",
                "dataset_file": "选择数据向量文件",
                "query_file": "选择查询向量文件",
                "enable_random": "启用内建随机数据集",
                "vec_volume": "向量总数",
                "vec_dim": "向量维度",
                "initial_size": "初始数据向量大小",
                "delete_size": "删除向量大小",
                "insert_size": "插入向量大小",
                "query_size": "查询向量大小",
                "annk": "查询的K值",
                "batch_size": "批量大小",
                "start": "开始",
                "progress": "当前步骤进度",
                "output": "中间输出",
                "result": "最终结果",
                "toggle_output": "显示/隐藏中间输出",
                "index_tag":"算法名称",
                "Event Rate":"事件速率",
                "cuda_idx":"Cuda设备号",
            }
        }
        self.current_language = "English"
        self.initUI()

    def initUI(self):
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)

        # Language selection
        self.language_selector = QComboBox()
        self.language_selector.addItems(self.languages.keys())
        self.language_selector.currentTextChanged.connect(self.update_language)
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
        self.configMap = None
        self.resultDic = None
       

        # Collapsible parameter section with form layout
        self.param_frame = QFrame()
        self.param_frame.setFrameShape(QFrame.StyledPanel)
        self.param_frame_layout = QFormLayout(self.param_frame)

        # Add a checkbox to enable or disable built-in random dataset
        self.enable_random_checkbox = QCheckBox(self.languages[self.current_language]["enable_random"])
        self.enable_random_checkbox.setChecked(True)
        self.initial_size_input = QLineEdit("50000")
        self.vec_volume_input = QLineEdit("100000")
        self.vec_dim_input = QLineEdit("768")
        self.delete_size_input = QLineEdit("0")
        self.insert_size_input = QLineEdit("50000")
        self.query_size_input = QLineEdit("100")
        self.annk_input = QLineEdit("10")
        self.event_rate_input = QLineEdit("-1")
        self.batch_size_input = QLineEdit("4000")
        self.idx_tag_input = QLineEdit("flat")
        self.cuda_input = QLineEdit("-1")
        self.param_frame_layout.addRow(self.enable_random_checkbox)
        self.param_frame_layout.addRow(self.languages[self.current_language]["index_tag"], self.idx_tag_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["vec_volume"], self.vec_volume_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["vec_dim"], self.vec_dim_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["event_rate"], self.event_rate_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["initial_size"], self.initial_size_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["delete_size"], self.delete_size_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["insert_size"], self.insert_size_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["query_size"], self.query_size_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["annk"], self.annk_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["batch_size"], self.batch_size_input)
        self.param_frame_layout.addRow(self.languages[self.current_language]["cuda_idx"], self.cuda_input)

        # Toggle button for parameter section
        self.toggle_button = QPushButton("Show/Hide Parameters")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.clicked.connect(self.toggle_parameters)

        # Add all widgets to main layout
        self.main_layout.addWidget(self.language_selector)
       
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.param_frame)
        # Progress bar
        self.progress_label = QLabel(self.languages[self.current_language]["progress"])
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        
         # Progress layout
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)

         # Collapsible Intermediate Output area
        self.toggle_output_button = QPushButton(self.languages[self.current_language]["toggle_output"])
        self.toggle_output_button.setCheckable(True)
        self.toggle_output_button.setChecked(True)
        self.toggle_output_button.clicked.connect(self.toggle_output_area)
        
        self.output_frame = QFrame()
        self.output_frame.setFrameShape(QFrame.StyledPanel)
        self.output_layout = QVBoxLayout(self.output_frame)
        

        self.output_label = QLabel(self.languages[self.current_language]["output"])
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        
        self.output_layout.addWidget(self.output_label)
        self.output_layout.addWidget(self.output_area)


        self.main_layout.addLayout(progress_layout)
        self.main_layout.addWidget(self.toggle_output_button)
        self.main_layout.addWidget(self.output_frame)

        self.save_result_button = QPushButton("Save Result")
        self.save_result_button.clicked.connect(self.save_result)

        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_result)

        # Add progress, output, and result to main layout
     # Intermediate output area changed to QTableWidget
        self.output_label = QLabel("Table Output")
        self.output_table = QTableWidget()  # Using QTableWidget for tabular display
        
        # Add to the layout where QTextEdit was used previously
        self.main_layout.addWidget(self.output_label)
        self.main_layout.addWidget(self.output_table)
       

        self.toggle_config_button = QPushButton("Config Preview/Result")
        #self.toggle_config_button.setCheckable(True)
        self.show_result =True
        self.toggle_config_button.clicked.connect(self.toggle_config)
        self.main_layout.addWidget(self.toggle_config_button)
        self.main_layout.addWidget(self.save_config_button)
        self.main_layout.addWidget(self.save_result_button)
        # Set central widget
        self.setCentralWidget(self.main_widget)
    def save_result(self):
        # 弹出文件保存对话框以选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Save Path", "", "All Files (*)")
        if file_path and (self.resultDic!= None):
            cfg = candy.dictToConfigMap(self.resultDic)
            cfg.toFile(file_path)
    def save_config(self):
        # 弹出文件保存对话框以选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Save Path", "", "All Files (*)")
        if file_path and (self.configMap!= None):
            cfg = candy.dictToConfigMap(self.configMap)
            cfg.toFile(file_path)
    def load_config_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Config File", "", "Config Files (*.csv *.yaml *.xml);;All Files (*)", options=options)
        if file_name:
            self.output_area.append(f"Loaded config file: {file_name}")
            configMap = candy.ConfigMap()
            configMap.fromFile(file_name)

            self.configMap = candy.configMapToDict(configMap)
            self.load_parameters_from_dict(self.configMap)
            self.output_label.setText("Table: Config preview")
            self.show_result =False
            self.displayConfigInTable(self.configMap)
    def toggle_output_area(self):
        self.output_frame.setVisible(self.toggle_output_button.isChecked())
    def toggle_config(self):
        self.show_result = not self.show_result
        if( not self.show_result):
            if(self.configMap!=None):
                self.displayConfigInTable(self.configMap)
                self.output_label.setText("Table: Config review")
        else:
            if(self.resultDic!=None):
                self.displayConfigInTable(self.resultDic)
                self.output_label.setText("Table: result")

    def update_language(self, lang):
        self.current_language = lang
        labels = self.languages[self.current_language]
        
        self.enable_random_checkbox.setText(labels["enable_random"])
        self.initial_size_input.setPlaceholderText(labels["initial_size"])
        self.delete_size_input.setPlaceholderText(labels["delete_size"])
        self.insert_size_input.setPlaceholderText(labels["insert_size"])
        self.query_size_input.setPlaceholderText(labels["query_size"])
        self.annk_input.setPlaceholderText(labels["annk"])
        self.batch_size_input.setPlaceholderText(labels["batch_size"])
        self.toggle_output_button.setText(labels["toggle_output"])
        self.output_label.setText(labels["result"])

        # Force refresh the visibility to update display
        is_visible = self.output_frame.isVisible()
        self.output_frame.setVisible(False)
        self.output_frame.setVisible(is_visible)

        # Force refresh the visibility to update display
        is_visible = self.param_frame.isVisible()
        self.param_frame.setVisible(False)
        self.param_frame.setVisible(is_visible)

    def select_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data Vector File")
        if file_path:
            self.data_file_path.setText(file_path)
    
    def select_query_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Query Vector File")
        if file_path:
            self.query_file_path.setText(file_path)

    def toggle_parameters(self):
        self.param_frame.setVisible(not self.toggle_button.isChecked())
    def get_parameters_dict(self):
        """
        从参数区域读取用户设置的值并生成一个 Python 字典。
        
        :return: dict - 包含参数区域设置的值
        """
        params = {
            "vecVolume":int(self.vec_volume_input.text()),
            "vecDim":int(self.vec_dim_input.text()),
            "initialRows": int(self.initial_size_input.text()),
            "deleteRows": int(self.delete_size_input.text()),
            "querySize": int(self.query_size_input.text()),
            "ANNK": int(self.annk_input.text()),
            "batchSize": int(self.batch_size_input.text()),
            "indexTag": (self.idx_tag_input.text()),
            "cudaDevice":int(self.cuda_input.text())
        }
        if(self.enable_random_checkbox.isChecked()):
            params['dataLoaderTag']=str('random')
        if(int(self.event_rate_input.text())<=0):
            params['staticDataSet']=1
        else:
            params['staticDataSet']=0
            params['eventRateTps']=int(self.event_rate_input.text())
        return params
    def load_parameters_from_dict(self, params):
        """
        从字典中加载参数值并更新到对应的编辑框和复选框。
        
        :param params: dict - 包含参数名称和值的字典
        """
        # 使用字典中的值更新对应的编辑框和复选框
        if "vecVolume" in params:
            self.vec_volume_input.setText(str(params["vecVolume"]))
        if "indexTag" in params:
            self.idx_tag_input.setText(str(params["indexTag"]))
        if "vecDim" in params:
            self.vec_dim_input.setText(str(params["vecDim"]))
        if "eventRateTps" in params:
            self.event_rate_input.setText(str(params["eventRateTps"]))
            if "staticDataSet" in params:
                if (int(str["staticDataSet"])==1):
                    self.event_rate_input.setText("-1")
        else:
            self.event_rate_input.setText("-1")
        if "initialRows" in params:
            self.initial_size_input.setText(str(params["initialRows"]))
        if "initialRows" in params:
            self.initial_size_input.setText(str(params["initialRows"]))
        if "deleteRows" in params:
            self.delete_size_input.setText(str(params["deleteRows"]))
        if "querySize" in params:
            self.query_size_input.setText(str(params["querySize"]))
        if "ANNK" in params:
            self.annk_input.setText(str(params["ANNK"]))
        if "batchSize" in params:
            self.batch_size_input.setText(str(params["batchSize"]))
        if "cudaDevice" in params:
            self.cuda_input.setText(str(params["cudaDevice"]))

    
    def displayConfigInTable(self, config_dict):
        from PyQt5.QtWidgets import QTableWidgetItem, QHeaderView

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

        # Adjust column width and row height to fill the entire table space
        self.output_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.output_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
        self.output_label.setText("Table: result")
        resultDic = self.thread.genCommonStatistics(recall)
        self.resultDic = resultDic
        self.displayConfigInTable(resultDic)
        self.show_result =True
    def start_benchmark(self):
        print('hello world')
        patchParams = self.get_parameters_dict()
        if self.configMap==None:
            self.configMap = patchParams
        else:
            self.configMap.update(patchParams)
        self.displayConfigInTable(self.configMap)
        self.configMap = self.benchmarkTool.loadConfigFromDict(self.configMap )
        self.refTool.loadConfigFromDict(self.configMap ,1,'flat')
        # 创建并启动后台线程
        self.thread = BenchmarkThread(self.benchmarkTool, self.configMap,windowObj=self)
        self.thread.progress_signal.connect(self.setProgressBar)
        self.thread.log_signal.connect(self.logOutput)
        self.thread.start()
        self.thread.finished.connect(self.on_anns_finished)  # 连接完成信号
# Run the application
app = QApplication(sys.argv)
window = BenchMarkMainWindow()
window.show()
sys.exit(app.exec_())
