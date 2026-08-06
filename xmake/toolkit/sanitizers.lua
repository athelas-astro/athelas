-- Sanitizer build modes.
--
-- Adds asan, tsan, ubsan, lsan, and allsan as ordinary build modes, mapped onto
-- xmake's target-level sanitizer policies. Each sets debug symbols and an
-- optimization level unless the target already chose one.
--
-- AddressSanitizer and ThreadSanitizer cannot be combined in one executable, so
-- allsan covers address, leak, and undefined behavior only. Run tsan separately.
--
-- Usage:
--
--   set_allowedmodes("debug", "release", "asan", "tsan", "ubsan", "lsan", "allsan")
--   add_rules("toolkit.sanitizers")
--
-- The rule applies to every target it is added to. Sanitizer instrumentation
-- reaches only targets built here, never prebuilt dependencies.

local sanitizers_for_mode = {
  asan = { "address" },
  tsan = { "thread" },
  ubsan = { "undefined" },
  lsan = { "leak" },
  allsan = { "address", "leak", "undefined" },
}

rule("toolkit.sanitizers")
after_load(function(target)
  local sanitizers
  for mode, names in pairs(sanitizers_for_mode) do
    if is_mode(mode) then
      sanitizers = names
      break
    end
  end
  if not sanitizers then
    return
  end

  if not target:get("symbols") then
    target:set("symbols", "debug")
  end
  if not target:get("optimize") then
    -- allsan stacks several sanitizers; unoptimized code keeps its reports
    -- readable. A single sanitizer stays usable at speed.
    target:set("optimize", is_mode("allsan") and "none" or "fastest")
  end
  for _, sanitizer in ipairs(sanitizers) do
    target:set("policy", "build.sanitizer." .. sanitizer, true)
  end
end)
rule_end()
