@fieldwise_init
struct RBERead(Copyable, Movable):
    var name: String
    var path: String


@fieldwise_init
struct VOICommand(Copyable, Movable):
    var name: String
    var action: String
    var tissue: String
    var weight_factor: Float64
    var max_dose_fraction: Float64
    var bin_state: String
    var argument0: Float64
    var argument1: Float64
    var use: Bool


@fieldwise_init
struct FieldSpec(Copyable, Movable):
    var number: Int
    var gantry_degrees: Float64
    var couch_degrees: Float64
    var raster_x_mm: Float64
    var raster_y_mm: Float64
    var contour_extension: Float64
    var z_step_mm: Float64
    var distal_extension_mm: Float64
    var robust_range_mm_h2o: Float64
    var robust_position_mm: Float64
    var max_threads: Int
    var target_voi: String
    var write_rst_path: String


@fieldwise_init
struct OptimizationSpec(Copyable, Movable):
    var enabled: Bool
    var biological: Bool
    var bioalg: String
    var optalg: String
    var dosealg: String
    var breakcrit: String
    var iterations: Int
    var grace_iterations: Int
    var eps: Float64
    var geps: Float64
    var chisquare_limit: Float64
    var diedose_limit: Float64
    var max_threads: Int
    var density_algorithm: String
    var dt_boundary_width_mm: Float64
    var no_preopt: Bool
    var no_dose_weight: Bool
    var complex_min_particles: Bool
    var selected_fields: String
    var ct_based: Bool
    var maximum_dose_weight: Float64


@fieldwise_init
struct DoseCalcSpec(Copyable, Movable):
    var enabled: Bool
    var output_base: String
    var algorithm: String
    var biological: Bool
    var bioalg: String
    var write: Bool
    var datatype: String
    var direct: Bool
    var voi: String
    var max_threads: Int
    var no_svv: Bool
    var no_rbe: Bool


@fieldwise_init
struct DVHSpec(Copyable, Movable):
    var enabled: Bool
    var output_base: String
    var biological: Bool
    var exchange_format: String
    var write: Bool


@fieldwise_init
struct ExecPlan(Copyable, Movable):
    var source_path: String
    var includes: List[String]
    var scancap_bolus_mm_h2o: Float64
    var scancap_off_h2o_mm: Float64
    var scancap_x_limit_mm: Float64
    var scancap_y_limit_mm: Float64
    var scancap_scanner_x_mm: Float64
    var scancap_scanner_y_mm: Float64
    var scancap_min_particles: Float64
    var scancap_couch_degrees: Float64
    var scancap_gantry_degrees: Float64
    var scancap_rifi_index: Int
    var sis_path: String
    var hlut_path: String
    var ddd_path_pattern: String
    var spc_path_pattern: String
    var dedx_path: String
    var rbe_tables: List[RBERead]
    var ct_base_path: String
    var voi_base_path: String
    var voi_read_selection: String
    var voi_commands: List[VOICommand]
    var prescription_dose_gy: Float64
    var target_tissue: String
    var residual_tissue: String
    var fields: List[FieldSpec]
    var has_optimization: Bool
    var optimization: OptimizationSpec
    var has_dose_calc: Bool
    var dose_calc: DoseCalcSpec
    var dose_output_base: String
    var has_dvh: Bool
    var dvh: DVHSpec
    var quit_requested: Bool


def parse_exec_file(path: String) raises -> ExecPlan:
    var plan = empty_exec_plan(path)
    parse_exec_file_lines(plan, path, False)
    var include_count = len(plan.includes)
    for i in range(include_count):
        var include_path = plan.includes[i].copy()
        parse_exec_file_lines(plan, include_path, True)
    return plan^


def parse_exec_file_lines(mut plan: ExecPlan, path: String, include_file: Bool) raises:
    with open(path, "r") as f:
        var lines = f.read().split("\n")
        for raw_line in lines:
            parse_exec_line(plan, strip_comment(String(raw_line)), include_file)


def print_exec_plan_summary(plan: ExecPlan):
    print("exec:", plan.source_path)
    print("included exec files:", len(plan.includes))
    print("ct base:", plan.ct_base_path)
    print("voi base:", plan.voi_base_path)
    print("scan-cap bolus mm H2O:", plan.scancap_bolus_mm_h2o)
    print("beamline off H2O mm:", plan.scancap_off_h2o_mm)
    print("scanner lateral limits:", plan.scancap_x_limit_mm, plan.scancap_y_limit_mm)
    print("hlut:", plan.hlut_path)
    print("rbe tables:", len(plan.rbe_tables))
    print("fields:", len(plan.fields))
    for i in range(len(plan.fields)):
        print(
            "field",
            plan.fields[i].number,
            "gantry",
            plan.fields[i].gantry_degrees,
            "couch",
            plan.fields[i].couch_degrees,
            "raster",
            plan.fields[i].raster_x_mm,
            plan.fields[i].raster_y_mm,
            "target",
            plan.fields[i].target_voi,
            "rst-write",
            plan.fields[i].write_rst_path,
        )
    print("optimization enabled:", plan.has_optimization)
    if plan.has_optimization:
        print("optimization algorithm:", plan.optimization.optalg)
        print("optimization dose algorithm:", plan.optimization.dosealg)
        print("optimization iterations:", plan.optimization.iterations)
    print("dose output base:", plan.dose_output_base)


def empty_exec_plan(path: String) -> ExecPlan:
    return ExecPlan(
        path,
        List[String](),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        "",
        "",
        "",
        "",
        "",
        List[RBERead](),
        "",
        "",
        "",
        List[VOICommand](),
        0.0,
        "",
        "",
        List[FieldSpec](),
        False,
        empty_optimization_spec(),
        False,
        empty_dose_calc_spec(),
        "",
        False,
        empty_dvh_spec(),
        False,
    )


def empty_field_spec(number: Int) -> FieldSpec:
    return FieldSpec(number, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "", "")


def empty_optimization_spec() -> OptimizationSpec:
    return OptimizationSpec(False, False, "", "", "", "", 0, 0, 0.0, 0.0, 0.0, 0.0, 0, "", 0.0, False, False, False, "", False, 0.0)


def empty_dose_calc_spec() -> DoseCalcSpec:
    return DoseCalcSpec(False, "", "", False, "", False, "", False, "", 0, False, False)


def empty_dvh_spec() -> DVHSpec:
    return DVHSpec(False, "", False, "", False)


def parse_exec_line(mut plan: ExecPlan, line: String, include_file: Bool) raises:
    var trimmed = strip(line)
    if trimmed.byte_length() == 0:
        return
    var command = first_token(trimmed)
    if command == "exec":
        validate_line_syntax(trimmed, ["exec"], List[String]())
        var include_path = first_quoted(trimmed)
        plan.includes.append(include_path)
        return
    if command == "scancap":
        validate_line_syntax(trimmed, ["scancap"], ["bolus", "offh2o", "x", "y", "scannerx", "scannery", "minparticles", "couch", "gantry", "rifi"])
        var parsed_bolus = option_float(trimmed, "bolus", plan.scancap_bolus_mm_h2o)
        if not include_file or plan.scancap_bolus_mm_h2o == 0.0:
            plan.scancap_bolus_mm_h2o = parsed_bolus
        plan.scancap_off_h2o_mm = option_float(trimmed, "offh2o", plan.scancap_off_h2o_mm)
        plan.scancap_x_limit_mm = option_float(trimmed, "x", plan.scancap_x_limit_mm)
        plan.scancap_y_limit_mm = option_float(trimmed, "y", plan.scancap_y_limit_mm)
        plan.scancap_scanner_x_mm = option_float(trimmed, "scannerx", plan.scancap_scanner_x_mm)
        plan.scancap_scanner_y_mm = option_float(trimmed, "scannery", plan.scancap_scanner_y_mm)
        plan.scancap_min_particles = option_float(trimmed, "minparticles", plan.scancap_min_particles)
        plan.scancap_couch_degrees = option_float(trimmed, "couch", plan.scancap_couch_degrees)
        plan.scancap_gantry_degrees = option_float(trimmed, "gantry", plan.scancap_gantry_degrees)
        plan.scancap_rifi_index = option_int(trimmed, "rifi", plan.scancap_rifi_index)
        return
    if command == "hlut":
        validate_line_syntax(trimmed, ["hlut", "read"], List[String]())
        plan.hlut_path = first_quoted(trimmed)
        return
    if command == "ddd":
        validate_line_syntax(trimmed, ["ddd", "read"], List[String]())
        plan.ddd_path_pattern = first_quoted(trimmed)
        return
    if command == "spc":
        validate_line_syntax(trimmed, ["spc", "read"], List[String]())
        plan.spc_path_pattern = first_quoted(trimmed)
        return
    if command == "dedx":
        validate_line_syntax(trimmed, ["dedx", "read"], List[String]())
        plan.dedx_path = first_quoted(trimmed)
        return
    if command == "sis":
        validate_line_syntax(trimmed, ["sis", "read"], List[String]())
        plan.sis_path = first_quoted(trimmed)
        return
    if command == "rbe":
        parse_rbe_line(plan, trimmed)
        return
    if command == "ct":
        validate_line_syntax(trimmed, ["ct", "read"], List[String]())
        plan.ct_base_path = first_quoted(trimmed)
        return
    if command == "voi":
        parse_voi_line(plan, trimmed)
        return
    if command == "plan":
        validate_line_syntax(trimmed, ["plan"], ["dose", "targettissue", "residualtissue"])
        plan.prescription_dose_gy = option_float(trimmed, "dose", plan.prescription_dose_gy)
        plan.target_tissue = option_string(trimmed, "targettissue", plan.target_tissue)
        plan.residual_tissue = option_string(trimmed, "residualtissue", plan.residual_tissue)
        return
    if command == "field":
        parse_field_line(plan, trimmed)
        return
    if command == "opt":
        parse_opt_line(plan, trimmed)
        return
    if command == "dose":
        parse_dose_line(plan, trimmed)
        return
    if command == "dvh":
        parse_dvh_line(plan, trimmed)
        return
    if command == "quit":
        validate_line_syntax(trimmed, ["quit"], List[String]())
        plan.quit_requested = True
        return
    raise Error("Unsupported .exec command: " + command)


def parse_rbe_line(mut plan: ExecPlan, line: String) raises:
    validate_line_syntax(line, ["rbe", "read"], ["alias"])
    var path_or_name = first_quoted(line)
    if contains_word(line, "read"):
        plan.rbe_tables.append(RBERead(basename_without_suffix(path_or_name), path_or_name))
    elif contains_option(line, "alias"):
        plan.voi_commands.append(VOICommand(path_or_name, "rbe-alias", option_string(line, "alias", ""), 0.0, 0.0, "", 0.0, 0.0, False))
    else:
        raise Error("rbe command must use read or alias")


def parse_voi_line(mut plan: ExecPlan, line: String) raises:
    validate_line_syntax(line, ["voi", "read", "use", "targetset", "oarset"], ["select", "binstate", "weightfactor", "maxdosefraction", "avoidance", "addmargin"])
    var name = first_quoted(line)
    if contains_word(line, "read"):
        plan.voi_base_path = name
        plan.voi_read_selection = option_string(line, "select", plan.voi_read_selection)
        return
    var action = ""
    if contains_word(line, "targetset"):
        action = "targetset"
    elif contains_word(line, "oarset"):
        action = "oarset"
    elif contains_option(line, "avoidance"):
        action = "avoidance"
    elif contains_word(line, "use"):
        action = "use"
    elif contains_option(line, "addmargin"):
        action = "addmargin"
    if action.byte_length() == 0 and not contains_word(line, "use"):
        raise Error("voi command has no supported action")
    var arguments = option_tuple2(line, "avoidance")
    if contains_option(line, "addmargin"):
        arguments.x = option_float(line, "addmargin", 0.0)
    plan.voi_commands.append(VOICommand(name, action, "", option_float(line, "weightfactor", 0.0), option_float(line, "maxdosefraction", 0.0), option_string(line, "binstate", ""), arguments.x, arguments.y, contains_word(line, "use")))


def parse_field_line(mut plan: ExecPlan, line: String) raises:
    validate_line_syntax(line, ["field", "new", "write"], ["couch", "gantry", "raster", "contourex", "zstep", "distal", "maxthreads", "robustrange", "robustpos", "settargetvoi", "file"])
    var parts = line.split()
    if len(parts) < 2:
        raise Error("field command is missing field number")
    var number = Int(parts[1])
    var index = field_index(plan, number)
    if index < 0:
        plan.fields.append(empty_field_spec(number))
        index = len(plan.fields) - 1
    if contains_word(line, "new"):
        plan.fields[index].couch_degrees = option_float(line, "couch", plan.fields[index].couch_degrees)
        plan.fields[index].gantry_degrees = option_float(line, "gantry", plan.fields[index].gantry_degrees)
        var raster = option_tuple2(line, "raster")
        plan.fields[index].raster_x_mm = raster.x
        plan.fields[index].raster_y_mm = raster.y
        plan.fields[index].contour_extension = option_float(line, "contourex", plan.fields[index].contour_extension)
        plan.fields[index].z_step_mm = option_float(line, "zstep", plan.fields[index].z_step_mm)
        plan.fields[index].distal_extension_mm = option_float(line, "distal", plan.fields[index].distal_extension_mm)
        plan.fields[index].max_threads = option_int(line, "maxthreads", plan.fields[index].max_threads)
        plan.fields[index].robust_range_mm_h2o = option_float(line, "robustrange", plan.fields[index].robust_range_mm_h2o)
        plan.fields[index].robust_position_mm = option_float(line, "robustpos", plan.fields[index].robust_position_mm)
        plan.fields[index].target_voi = option_string(line, "settargetvoi", plan.fields[index].target_voi)
    elif contains_word(line, "write"):
        plan.fields[index].write_rst_path = option_string(line, "file", plan.fields[index].write_rst_path)
    else:
        raise Error("field command must use new or write")


def parse_opt_line(mut plan: ExecPlan, line: String) raises:
    validate_line_syntax(line, ["opt", "bio", "phys", "ctbased", "nopreopt", "nodoseweight", "complexminp"], ["field", "breakcrit", "bioalg", "optalg", "dosealg", "graceiter", "i", "densityalg", "setDTBoundaryWidth", "diedoselim", "eps", "geps", "chisquarelimit", "maxthreads", "maxdosew"])
    plan.has_optimization = True
    plan.optimization.enabled = True
    plan.optimization.biological = contains_word(line, "bio")
    plan.optimization.bioalg = option_string(line, "bioalg", plan.optimization.bioalg)
    plan.optimization.optalg = option_string(line, "optalg", plan.optimization.optalg)
    plan.optimization.dosealg = option_string(line, "dosealg", plan.optimization.dosealg)
    plan.optimization.breakcrit = option_string(line, "breakcrit", plan.optimization.breakcrit)
    plan.optimization.iterations = option_int(line, "i", plan.optimization.iterations)
    plan.optimization.grace_iterations = option_int(line, "graceiter", plan.optimization.grace_iterations)
    plan.optimization.eps = option_float(line, "eps", plan.optimization.eps)
    plan.optimization.geps = option_float(line, "geps", plan.optimization.geps)
    plan.optimization.chisquare_limit = option_float(line, "chisquarelimit", plan.optimization.chisquare_limit)
    plan.optimization.diedose_limit = option_float(line, "diedoselim", plan.optimization.diedose_limit)
    plan.optimization.max_threads = option_int(line, "maxthreads", plan.optimization.max_threads)
    plan.optimization.density_algorithm = option_string(line, "densityalg", plan.optimization.density_algorithm)
    plan.optimization.dt_boundary_width_mm = option_float(line, "setDTBoundaryWidth", plan.optimization.dt_boundary_width_mm)
    plan.optimization.no_preopt = contains_word(line, "nopreopt")
    plan.optimization.no_dose_weight = contains_word(line, "nodoseweight")
    plan.optimization.complex_min_particles = contains_word(line, "complexminp")
    plan.optimization.selected_fields = option_string(line, "field", plan.optimization.selected_fields)
    plan.optimization.ct_based = contains_word(line, "ctbased")
    plan.optimization.maximum_dose_weight = option_float(line, "maxdosew", plan.optimization.maximum_dose_weight)


def parse_dose_line(mut plan: ExecPlan, line: String) raises:
    validate_line_syntax(line, ["dose", "calc", "bio", "phys", "nosvv", "norbe", "write", "direct"], ["alg", "bioalg", "datatype", "voi", "maxthreads"])
    plan.has_dose_calc = True
    plan.dose_calc.enabled = True
    plan.dose_calc.output_base = first_quoted(line)
    plan.dose_output_base = plan.dose_calc.output_base
    plan.dose_calc.algorithm = option_string(line, "alg", plan.dose_calc.algorithm)
    plan.dose_calc.biological = contains_word(line, "bio")
    plan.dose_calc.bioalg = option_string(line, "bioalg", plan.dose_calc.bioalg)
    plan.dose_calc.write = contains_word(line, "write")
    plan.dose_calc.datatype = option_string(line, "datatype", plan.dose_calc.datatype)
    plan.dose_calc.direct = contains_word(line, "direct")
    plan.dose_calc.voi = option_string(line, "voi", plan.dose_calc.voi)
    plan.dose_calc.max_threads = option_int(line, "maxthreads", plan.dose_calc.max_threads)
    plan.dose_calc.no_svv = contains_word(line, "nosvv")
    plan.dose_calc.no_rbe = contains_word(line, "norbe")


def parse_dvh_line(mut plan: ExecPlan, line: String) raises:
    validate_line_syntax(line, ["dvh", "calc", "bio", "write"], ["ex"])
    plan.has_dvh = True
    plan.dvh.enabled = contains_word(line, "calc")
    plan.dvh.output_base = first_quoted(line)
    plan.dvh.biological = contains_word(line, "bio")
    plan.dvh.exchange_format = option_string(line, "ex", "")
    plan.dvh.write = contains_word(line, "write")


def require_standalone_setup_only(plan: ExecPlan) raises:
    if plan.has_optimization or plan.has_dose_calc or plan.has_dvh:
        raise Error("standalone Mojo .exec execution is intentionally unsupported; use the TRiP C control plane and packed Mojo backends")
    for field in plan.fields:
        if field.write_rst_path.byte_length() > 0:
            raise Error("standalone Mojo does not write RST output; use the TRiP C control plane")


def field_index(plan: ExecPlan, number: Int) -> Int:
    for i in range(len(plan.fields)):
        if plan.fields[i].number == number:
            return i
    return -1


@fieldwise_init
struct FloatPair(Copyable, Movable):
    var x: Float64
    var y: Float64


def option_tuple2(line: String, name: String) raises -> FloatPair:
    var body = option_body(line, name)
    var parts = body.split(",")
    if len(parts) >= 2:
        return FloatPair(Float64(parts[0]), Float64(parts[1]))
    return FloatPair(0.0, 0.0)


def option_float(line: String, name: String, fallback: Float64) raises -> Float64:
    var body = option_body(line, name)
    if body.byte_length() == 0:
        return fallback
    return Float64(body)


def option_int(line: String, name: String, fallback: Int) raises -> Int:
    var body = option_body(line, name)
    if body.byte_length() == 0:
        return fallback
    return Int(body)


def option_string(line: String, name: String, fallback: String) -> String:
    var body = option_body(line, name)
    if body.byte_length() == 0:
        return fallback
    return body^


def option_body(line: String, name: String) -> String:
    var needle = name + "("
    var start = find_option_start(line, needle)
    if start < 0:
        return ""
    start += needle.byte_length()
    var depth = 1
    var out = String()
    for i in range(start, line.byte_length()):
        var c = line[byte=i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return out^
        out += String(c)
    return out^


def find_option_start(value: String, needle: String) -> Int:
    if needle.byte_length() == 0:
        return 0
    if needle.byte_length() > value.byte_length():
        return -1
    for i in range(value.byte_length() - needle.byte_length() + 1):
        if i > 0 and is_option_name_char(value[byte=i - 1]):
            continue
        var ok = True
        for j in range(needle.byte_length()):
            if value[byte=i + j] != needle[byte=j]:
                ok = False
                break
        if ok:
            return i
    return -1


def is_option_name_char(c: StringSlice) -> Bool:
    return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or (c >= "0" and c <= "9") or c == "_"


def contains_option(line: String, name: String) -> Bool:
    return find_substring(line, name + "(") >= 0


def contains_word(line: String, word: String) -> Bool:
    var parts = line.split()
    for i in range(len(parts)):
        var token = strip_punctuation(String(parts[i]))
        if token == word:
            return True
    return False


def validate_line_syntax(
    line: String, allowed_words: List[String], allowed_options: List[String]
) raises:
    var quoted = False
    var depth = 0
    var index = 0
    while index < line.byte_length():
        var c = line[byte=index]
        if c == "\"":
            quoted = not quoted
            index += 1
            continue
        if quoted:
            index += 1
            continue
        if c == "(":
            depth += 1
            index += 1
            continue
        if c == ")":
            depth -= 1
            if depth < 0:
                raise Error("unbalanced .exec option parentheses")
            index += 1
            continue
        if depth == 0 and is_option_name_char(c) and not (
            c >= "0" and c <= "9"
        ):
            var start = index
            index += 1
            while index < line.byte_length() and is_option_name_char(
                line[byte=index]
            ):
                index += 1
            var name = substring(line, start, index)
            var is_option = index < line.byte_length() and line[byte=index] == "("
            if is_option:
                if not string_list_contains(allowed_options, name):
                    raise Error("unsupported .exec option: " + name)
            elif not string_list_contains(allowed_words, name):
                raise Error("unsupported .exec flag or action: " + name)
            continue
        index += 1
    if quoted:
        raise Error("unterminated quote in .exec command")
    if depth != 0:
        raise Error("unbalanced .exec option parentheses")


def string_list_contains(values: List[String], expected: String) -> Bool:
    for value in values:
        if value == expected:
            return True
    return False


def first_token(line: String) -> String:
    var parts = line.split()
    if len(parts) == 0:
        return ""
    return String(parts[0])


def first_quoted(line: String) -> String:
    var active = False
    var out = String()
    for i in range(line.byte_length()):
        var c = line[byte=i]
        if c == "\"":
            if active:
                return out^
            active = True
        elif active:
            out += String(c)
    return out^


def strip_comment(line: String) -> String:
    var out = String()
    var quoted = False
    for i in range(line.byte_length()):
        var c = line[byte=i]
        if c == "\"":
            quoted = not quoted
        if not quoted and c == "#":
            return strip(out)
        out += String(c)
    return strip(out)


def strip(value: String) -> String:
    var start = 0
    var end = value.byte_length()
    while start < end and is_space(value[byte=start]):
        start += 1
    while end > start and is_space(value[byte=end - 1]):
        end -= 1
    return substring(value, start, end)


def strip_punctuation(value: String) -> String:
    var end = value.byte_length()
    while end > 0:
        var c = value[byte=end - 1]
        if String(c) != "/" and String(c) != ",":
            break
        end -= 1
    return substring(value, 0, end)


def is_space(c: StringSlice) -> Bool:
    return c == " " or c == "\t" or c == "\r" or c == "\n"


def substring(value: String, start: Int, end: Int) -> String:
    var out = String()
    for i in range(start, end):
        out += String(value[byte=i])
    return out^


def find_substring(value: String, needle: String) -> Int:
    if needle.byte_length() == 0:
        return 0
    if needle.byte_length() > value.byte_length():
        return -1
    for i in range(value.byte_length() - needle.byte_length() + 1):
        var ok = True
        for j in range(needle.byte_length()):
            if value[byte=i + j] != needle[byte=j]:
                ok = False
                break
        if ok:
            return i
    return -1


def basename_without_suffix(path: String) -> String:
    var base_start = 0
    for i in range(path.byte_length()):
        if path[byte=i] == "/":
            base_start = i + 1
    var base_end = path.byte_length()
    for i in range(base_start, path.byte_length()):
        if path[byte=i] == ".":
            base_end = i
            break
    return substring(path, base_start, base_end)
