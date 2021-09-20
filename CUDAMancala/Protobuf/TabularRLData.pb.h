// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: TabularRLData.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_TabularRLData_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_TabularRLData_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3017000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3017003 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_TabularRLData_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_TabularRLData_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_TabularRLData_2eproto;
namespace mancala {
class QAgent;
struct QAgentDefaultTypeInternal;
extern QAgentDefaultTypeInternal _QAgent_default_instance_;
}  // namespace mancala
PROTOBUF_NAMESPACE_OPEN
template<> ::mancala::QAgent* Arena::CreateMaybeMessage<::mancala::QAgent>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mancala {

// ===================================================================

class QAgent final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mancala.QAgent) */ {
 public:
  inline QAgent() : QAgent(nullptr) {}
  ~QAgent() override;
  explicit constexpr QAgent(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  QAgent(const QAgent& from);
  QAgent(QAgent&& from) noexcept
    : QAgent() {
    *this = ::std::move(from);
  }

  inline QAgent& operator=(const QAgent& from) {
    CopyFrom(from);
    return *this;
  }
  inline QAgent& operator=(QAgent&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const QAgent& default_instance() {
    return *internal_default_instance();
  }
  static inline const QAgent* internal_default_instance() {
    return reinterpret_cast<const QAgent*>(
               &_QAgent_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(QAgent& a, QAgent& b) {
    a.Swap(&b);
  }
  inline void Swap(QAgent* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(QAgent* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline QAgent* New() const final {
    return new QAgent();
  }

  QAgent* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<QAgent>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const QAgent& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const QAgent& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message*to, const ::PROTOBUF_NAMESPACE_ID::Message&from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(QAgent* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mancala.QAgent";
  }
  protected:
  explicit QAgent(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kQFieldNumber = 3,
    kNpitsFieldNumber = 1,
    kNseedsFieldNumber = 2,
  };
  // repeated float Q = 3;
  int q_size() const;
  private:
  int _internal_q_size() const;
  public:
  void clear_q();
  private:
  float _internal_q(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_q() const;
  void _internal_add_q(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_q();
  public:
  float q(int index) const;
  void set_q(int index, float value);
  void add_q(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      q() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_q();

  // int32 npits = 1;
  void clear_npits();
  ::PROTOBUF_NAMESPACE_ID::int32 npits() const;
  void set_npits(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_npits() const;
  void _internal_set_npits(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // int32 nseeds = 2;
  void clear_nseeds();
  ::PROTOBUF_NAMESPACE_ID::int32 nseeds() const;
  void set_nseeds(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_nseeds() const;
  void _internal_set_nseeds(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // @@protoc_insertion_point(class_scope:mancala.QAgent)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > q_;
  ::PROTOBUF_NAMESPACE_ID::int32 npits_;
  ::PROTOBUF_NAMESPACE_ID::int32 nseeds_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_TabularRLData_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// QAgent

// int32 npits = 1;
inline void QAgent::clear_npits() {
  npits_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 QAgent::_internal_npits() const {
  return npits_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 QAgent::npits() const {
  // @@protoc_insertion_point(field_get:mancala.QAgent.npits)
  return _internal_npits();
}
inline void QAgent::_internal_set_npits(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  npits_ = value;
}
inline void QAgent::set_npits(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_npits(value);
  // @@protoc_insertion_point(field_set:mancala.QAgent.npits)
}

// int32 nseeds = 2;
inline void QAgent::clear_nseeds() {
  nseeds_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 QAgent::_internal_nseeds() const {
  return nseeds_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 QAgent::nseeds() const {
  // @@protoc_insertion_point(field_get:mancala.QAgent.nseeds)
  return _internal_nseeds();
}
inline void QAgent::_internal_set_nseeds(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  nseeds_ = value;
}
inline void QAgent::set_nseeds(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_nseeds(value);
  // @@protoc_insertion_point(field_set:mancala.QAgent.nseeds)
}

// repeated float Q = 3;
inline int QAgent::_internal_q_size() const {
  return q_.size();
}
inline int QAgent::q_size() const {
  return _internal_q_size();
}
inline void QAgent::clear_q() {
  q_.Clear();
}
inline float QAgent::_internal_q(int index) const {
  return q_.Get(index);
}
inline float QAgent::q(int index) const {
  // @@protoc_insertion_point(field_get:mancala.QAgent.Q)
  return _internal_q(index);
}
inline void QAgent::set_q(int index, float value) {
  q_.Set(index, value);
  // @@protoc_insertion_point(field_set:mancala.QAgent.Q)
}
inline void QAgent::_internal_add_q(float value) {
  q_.Add(value);
}
inline void QAgent::add_q(float value) {
  _internal_add_q(value);
  // @@protoc_insertion_point(field_add:mancala.QAgent.Q)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
QAgent::_internal_q() const {
  return q_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
QAgent::q() const {
  // @@protoc_insertion_point(field_list:mancala.QAgent.Q)
  return _internal_q();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
QAgent::_internal_mutable_q() {
  return &q_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
QAgent::mutable_q() {
  // @@protoc_insertion_point(field_mutable_list:mancala.QAgent.Q)
  return _internal_mutable_q();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mancala

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_TabularRLData_2eproto
