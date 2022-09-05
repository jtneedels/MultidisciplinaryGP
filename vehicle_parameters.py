
class VehicleParameters:
	""" Object holding all vehicle parameters.

    """
	def __init__(self):
		""" Initializer method.

		Args:
			l_d: max lift/drag ratio
			b: ballistic coefficient kg/m2
			length: max axial length m
			rn: leading edge radius m

    	"""
		self.l_d = 2.562
		self.b = 2992
		self.length = 1
		self.rn = 0.034

	class TpsParameters:
		""" Object holding thermal protection system parameters.

    	"""
		def __init__(self):
			""" Initializer method.

			Args:
				k: thermnal conductivity W/m-K
				rho: density kg/m^3
				cp: J/kg-K

			"""
			self.k = 0.2
			self.rho = 1205
			self.cp = 194

	class StructuralParameters:
		""" Object holding structural parameters.

		"""
		def __init__(self):
			""" Initializer method.

			Args:


			"""
