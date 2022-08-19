import sys
import numpy
import numpy as np
sys.path.append("D:/Artigence/BCI")
print(sys.path)
import Train_and_class
from Train_and_class import Network_class
import Write

#saving result path




class MyOVBox(OVBox):

	def __init__(self):
		OVBox.__init__(self)
		self.signalHeader = None
		self.signalBuffer = list()
		self.network_input = list()
		self.count = 0
		self.signal = None
		self.NC = Network_class()
	def process(self):
		for chunkIdx in range(len(self.input[0])):
			#print(chunkIdx)

			# Include signal's information
			if (type(self.input[0][chunkIdx]) == OVSignalHeader):
				self.signalHeader = self.input[0].pop()
				outputHeader = OVSignalHeader(self.signalHeader.startTime, self.signalHeader.endTime,
											  self.signalHeader.dimensionSizes,
											  self.signalHeader.dimensionLabels,
											  self.signalHeader.samplingRate)

				self.output[0].append(outputHeader)
			# save signal as nuppy array
			elif (type(self.input[0][chunkIdx]) == OVSignalBuffer):
				chunk = self.input[0].pop()
				
				numpyBuffer = numpy.array(chunk)
				#if (chunk.endTime>30):
					#print(numpyBuffer)
				# numpyBuffer = numpyBuffer.mean(axis=0)
				
				#print(chunk.startTime, chunk.endTime)
				chunk = OVSignalBuffer(chunk.startTime, chunk.endTime, numpyBuffer.tolist())
				
				
				# Save every 3 seconds
				if chunk.endTime % 3  == 0.:
			
					self.count = 0
					temp = []
					to_network = np.array(numpyBuffer)
					for i in range(32):
						temp.append(to_network[8*i:8*(i+1)])
					
					self.signalBuffer.append(temp)
					
					for i in range(len(self.signalBuffer)):
						self.signal = self.signalBuffer.pop(0)
						self.signal = numpy.array(self.signal)

						if i ==0:
							self.network_input = self.signal.tolist()
						else:
							self.network_input = np.concatenate((np.array(self.network_input),self.signal),axis = 1).tolist()
						
						self.signal = None
					# pass to network as input
					#print(len(self.network_input))
					self.NC.Network(self.network_input)
					
					self.network_input.clear()
				

				else:
					#print("in")
					temp = []
					to_network = np.array(numpyBuffer)
					#print(len(numpyBuffer))
					for i in range(32):
						temp.append(to_network[8*i:8*(i+1)])
					#print(np.array(temp).shape)
					self.signalBuffer.append(temp)
					#print(len(self.signalBuffer))
			
			
			elif (type(self.input[0][chunkIdx]) == OVSignalEnd):
				self.output[0].append(self.input[0].pop())
				

	def uninitialize(self):
		print('Python uninitialize function started')
		return
box = MyOVBox()
