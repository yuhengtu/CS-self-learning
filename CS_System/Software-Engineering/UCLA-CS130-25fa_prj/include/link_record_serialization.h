#ifndef LINK_RECORD_SERIALIZATION_H
#define LINK_RECORD_SERIALIZATION_H

#include <optional>
#include <string>

#include "link_manager_types.h"

// Serialize a LinkRecord to a JSON string for storage.
std::string LinkRecordToJson(const LinkRecord& rec);

// Serialize a LinkRecord to JSON including the code field (for API responses).
std::string LinkRecordToJsonWithCode(const LinkRecord& rec);

// Parse a stored JSON string into a LinkRecord, assigning the provided code.
// Returns std::nullopt if the JSON is malformed or required fields are missing.
std::optional<LinkRecord> LinkRecordFromJson(const std::string& json,
                                             const std::string& code);

#endif // LINK_RECORD_SERIALIZATION_H
