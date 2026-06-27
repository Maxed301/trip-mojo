@fieldwise_init
struct RSTPlan:
    var patient_id: String
    var projectile: Particle
    var gantry: Int
    var couch: Int
    var bolus: Int
    var ripplefilter: Int
    var field: List[RasterSpot]

    def __init__(out self, path: String) raises:
        self.patient_id = ""
        self.projectile = Particle("", 0, 0)
        self.gantry = 0
        self.couch = 0
        self.bolus = 0
        self.ripplefilter = 0
        self.field = List[RasterSpot]()
        var energy = 0.0
        var focus = 0.0
        var range_mm_h2o = 0.0
        var remaining = 0
        with open(path, "r") as f:
            var content = f.read()
            var lines = content.split("\n")
            for line in lines:
                if not line:
                    continue
                parts = line.split()
                if parts[0] == 'patient_id':
                    self.patient_id = String(parts[1])
                elif parts[0] == 'projectile':
                    self.projectile.name = String(parts[1])
                elif parts[0] == 'charge':
                    self.projectile.charge = Int(parts[1])
                elif parts[0] == 'mass':
                    self.projectile.mass = Int(parts[1])
                elif parts[0] == 'gantryangle':
                    self.gantry= Int(parts[1])
                elif parts[0] == 'couchangle':
                    self.couch = Int(parts[1])
                elif parts[0] == 'bolus':
                    self.bolus = Int(parts[1])
                elif parts[0] == 'ripplefilter':
                    self.ripplefilter = Int(parts[1])
                elif parts[0] == 'submachine#':
                    range_mm_h2o = Float64(parts[1])
                    energy = Float64(parts[2])
                    focus = Float64(parts[4])
                    remaining = 0
                elif parts[0] == '#points':
                    remaining = Int(parts[1])
                elif remaining > 0:
                    self.field.append(RasterSpot(Float64(parts[0]), Float64(parts[1]), energy, Float64(parts[2]), focus, range_mm_h2o, 0.0, True, len(self.field)))
                    remaining -= 1

@fieldwise_init
struct Particle(Copyable, Movable):
    var name: String
    var charge: Int
    var mass: Int


struct RasterSpot(Copyable, Movable):
    var x: Float64
    var y: Float64
    var energy: Float64
    var particles: Float64
    var focus: Float64
    var range_mm_h2o: Float64
    var f2_max: Float64
    var active: Bool
    var point_index: Int

    def __init__(
        out self,
        x: Float64,
        y: Float64,
        energy: Float64,
        particles: Float64,
        focus: Float64,
        range_mm_h2o: Float64 = 0.0,
        f2_max: Float64 = 0.0,
        active: Bool = True,
        point_index: Int = -1,
    ):
        self.x = x
        self.y = y
        self.energy = energy
        self.particles = particles
        self.focus = focus
        self.range_mm_h2o = range_mm_h2o
        self.f2_max = f2_max
        self.active = active
        self.point_index = point_index
